import torch
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import json
import math
import os
import time
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten # to be continued
from evaluator.compute_metrics import compute_metrics, DATASET_TO_OUTPUT_LANG
from transformers import GenerationConfig

class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_task_list,
                 eval_task_list,
                 test_task_list,
                 args):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_task_list = train_task_list
        self.eval_task_list = eval_task_list
        self.test_task_list = test_task_list
        self.args = args
        self.generation_config = GenerationConfig(
            do_sample=self.args.do_sample,
            temperature=self.args.temperature if self.args.do_sample else None,
            top_p=self.args.top_p if self.args.do_sample else None,
            repetition_penalty=self.args.repetition_penalty,
        )

    def perplexity_evaluation(self, eval_dataloader, device):
        # 验证集上测困惑度
        self.model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    def token_weighted_validation_nll(self, eval_dataloader, device):
        """Return answer-token-weighted validation NLL.

        Hugging Face causal-LM loss is already averaged over unmasked target
        tokens for each batch. Multiplying it by the number of target tokens
        before reducing avoids giving short and long batches equal weight.
        """
        was_training = self.model.training
        self.model.eval()
        nll_sum = torch.zeros(1, dtype=torch.float64, device=device)
        token_count = torch.zeros(1, dtype=torch.float64, device=device)

        for batch in eval_dataloader:
            batch = dict(batch)
            batch.pop('sources', None)
            batch.pop('gts', None)
            batch = to_device(batch, device)

            # Causal-LM loss shifts labels by one position internally.
            num_target_tokens = (batch['labels'][..., 1:] != -100).sum()
            if num_target_tokens.item() == 0:
                continue

            with torch.no_grad():
                outputs = self.model(**batch, use_cache=False)
            nll_sum += outputs.loss.detach().double() * num_target_tokens.double()
            token_count += num_target_tokens.double()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(nll_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

        if was_training:
            self.model.train()

        if token_count.item() == 0:
            raise RuntimeError("Validation set contains no unmasked answer tokens.")
        return (nll_sum / token_count).item(), int(token_count.item())

    def _global_sum_int(self, value, device):
        value_tensor = torch.tensor([value], dtype=torch.long, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        return int(value_tensor.item())

    def _log_convergence_record(
        self,
        task,
        event,
        optimizer_step,
        epoch,
        train_loss,
        local_target_tokens_seen,
        training_seconds,
        eval_dataloader,
        device,
    ):
        eval_started = time.perf_counter()
        validation_nll, validation_tokens = self.token_weighted_validation_nll(
            eval_dataloader,
            device,
        )
        evaluation_seconds = time.perf_counter() - eval_started
        target_tokens_seen = self._global_sum_int(local_target_tokens_seen, device)

        record = {
            "task": task,
            "source_adapter": getattr(self.args, "init_lora_path", None) or "fresh",
            "seed": int(self.args.seed),
            "event": event,
            "epoch": int(epoch),
            "optimizer_step": int(optimizer_step),
            "target_tokens_seen": target_tokens_seen,
            "train_loss": None if train_loss is None else float(train_loss),
            "validation_nll": float(validation_nll),
            "validation_perplexity": float(math.exp(min(validation_nll, 20.0))),
            "validation_tokens": validation_tokens,
            "training_seconds": float(training_seconds),
            "evaluation_seconds": float(evaluation_seconds),
        }

        if self.args.global_rank == 0:
            if self.args.output_dir is None:
                raise ValueError("--output_dir is required for convergence logging.")
            os.makedirs(self.args.output_dir, exist_ok=True)
            output_path = os.path.join(self.args.output_dir, "convergence.jsonl")
            with open(output_path, "a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print_rank_0(
                "[convergence] "
                f"event={event} task={task} optimizer_step={optimizer_step} "
                f"target_tokens={target_tokens_seen} val_nll={validation_nll:.6f} "
                f"train_seconds={training_seconds:.2f}",
                self.args.global_rank,
            )
        return record

    def _task_eval_from_predictions(self, task, sources_sequences, predicted_sequences, ground_truths):
        if task in ['CodeSearchNet', 'TheVault_Csharp']:
            calc_codebleu = False
        else:
            calc_codebleu = True
        return compute_metrics(predicted_sequences, ground_truths, calc_codebleu=calc_codebleu, language=DATASET_TO_OUTPUT_LANG.get(task, None))

    def task_generation_evaluation(self, task, test_dataloader, device, max_ans_len=None, return_predictions=False):
        self.model.eval()
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []

        if max_ans_len is None:
            max_ans_len = getattr(self.args, "max_ans_len", 256)

        is_executable = getattr(self.args, "benchmark", "non-executable") != "non-executable"
        if is_executable:
            return_predictions = True
            num_return_sequences = int(getattr(self.args, "num_return_sequences", 1))
            top_k = int(getattr(self.args, "top_k", 0))
            generation_kwargs = self.generation_config.to_dict()
            generation_kwargs.update({
                "num_return_sequences": num_return_sequences,
                "top_k": top_k,
            })
            generation_config = GenerationConfig(**generation_kwargs)
        else:
            num_return_sequences = 1
            generation_config = self.generation_config

        progress_bar = tqdm(total=len(test_dataloader), leave=True, disable=(self.args.global_rank != 0))
        for step, batch in enumerate(test_dataloader):
            sources_sequences += batch['sources']
            if 'gts' in batch:
                ground_truths += batch['gts']
                del batch['gts']
            elif 'labels' in batch:
                label_tensor = batch['labels']
                for row in label_tensor:
                    valid_ids = row[row != -100].detach().cpu().tolist()
                    ground_truths.append(
                        self.tokenizer.decode(valid_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    )
                del batch['labels']
            else:
                ground_truths += [''] * len(batch['sources'])

            del batch['sources']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]

            with torch.no_grad():
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.tokenizer.eos_token_id

                generate_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=max_ans_len,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    use_cache=True,
                )

            sequences = self.tokenizer.batch_decode(
                generate_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            if is_executable and num_return_sequences > 1:
                batch_preds = [
                    sequences[i:i + num_return_sequences]
                    for i in range(0, len(sequences), num_return_sequences)
                ]
                predicted_sequences.extend(batch_preds)
            else:
                predicted_sequences += sequences

            if self.args.global_rank == 0:
                progress_bar.update(1)
                description = f"Test step {step}"
                progress_bar.set_description(description, refresh=False)

        metrics = {} if is_executable else self._task_eval_from_predictions(
            task, sources_sequences, predicted_sequences, ground_truths
        )
        if return_predictions:
            prediction_rows = [
                {
                    "source": source,
                    "ground-truth": gt,
                    "prediction": pred,
                }
                for source, gt, pred in zip(sources_sequences, ground_truths, predicted_sequences)
            ]
            return metrics, prediction_rows
        return metrics

    def _resolve_max_ans_len(self, task_idx):
        max_ans_len = getattr(self.args, "max_ans_len", 256)
        if isinstance(max_ans_len, (list, tuple)):
            if len(max_ans_len) == 0:
                return 256
            if len(max_ans_len) == 1:
                return int(max_ans_len[0])
            return int(max_ans_len[task_idx])
        return int(max_ans_len)

    def _save_generation_predictions(self, split_name, task_idx, task, metrics, prediction_rows):
        if self.args.global_rank != 0 or self.args.output_dir is None:
            return
        safe_task_name = str(task).replace("/", "_").replace(":", "_")
        pred_dir = os.path.join(self.args.output_dir, "predictions", split_name)
        os.makedirs(pred_dir, exist_ok=True)
        pred_file = os.path.join(pred_dir, f"{task_idx}_{safe_task_name}.json")
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "predictions": prediction_rows}, f, ensure_ascii=False, indent=2)
        print_rank_0(f"Saved {split_name} predictions to {pred_file}", self.args.global_rank)

    def evaluate_seen_tasks_after_training(self, trained_task, trained_task_idx, device):
        split_name = f"test-after-task-{trained_task_idx}"
        for seen_idx, (seen_task, test_dataloader) in enumerate(list(self.test_task_list.items())[:trained_task_idx + 1]):
            print_rank_0(
                f"***** Testing on seen task {seen_task} after training task {trained_task} *****",
                self.args.global_rank,
            )
            test_result, test_predictions = self.task_generation_evaluation(
                seen_task,
                test_dataloader,
                device,
                max_ans_len=self._resolve_max_ans_len(seen_idx),
                return_predictions=True,
            )
            print_rank_0(
                f"[seen-task={seen_task} after-task={trained_task}] test result: {test_result}",
                self.args.global_rank,
            )
            self._save_generation_predictions(split_name, seen_idx, seen_task, test_result, test_predictions)

    def test_all_tasks_and_save_predictions(self):
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)

        prediction_root = os.path.join(self.args.output_dir or ".", "predictions", f"final-{self.__class__.__name__}")
        if self.args.global_rank == 0:
            os.makedirs(prediction_root, exist_ok=True)

        if getattr(self.args, "benchmark", "non-executable") == "non-executable":
            final_metrics = {}
            for task_idx, (task_name, test_dataloader) in enumerate(self.test_task_list.items()):
                print_rank_0(
                    f"***** Final testing on task {task_name} after continual training *****",
                    self.args.global_rank,
                )
                test_result, prediction_rows = self.task_generation_evaluation(
                    task_name,
                    test_dataloader,
                    device,
                    max_ans_len=self._resolve_max_ans_len(task_idx),
                    return_predictions=True,
                )
                final_metrics[task_name] = test_result
                print_rank_0(f"[final-test task={task_name}] result: {test_result}", self.args.global_rank)

                if self.args.global_rank == 0:
                    safe_task_name = str(task_name).replace("/", "_").replace(":", "_")
                    prediction_file = os.path.join(prediction_root, f"{task_idx}_{safe_task_name}.json")
                    with open(prediction_file, "w", encoding="utf-8") as f:
                        json.dump(prediction_rows, f, ensure_ascii=False, indent=2)
                    print_rank_0(f"Saved final-test predictions to {prediction_file}", self.args.global_rank)

            if self.args.global_rank == 0:
                metrics_file = os.path.join(prediction_root, "metrics_summary.json")
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump(final_metrics, f, ensure_ascii=False, indent=2)
                print_rank_0(f"Saved final-test metrics to {metrics_file}", self.args.global_rank)
        else:
            for task_idx, (task_name, test_dataloader) in enumerate(self.test_task_list.items()):
                print_rank_0(
                    f"***** Final testing on task {task_name} after continual training *****",
                    self.args.global_rank,
                )
                _, prediction_rows = self.task_generation_evaluation(
                    task_name,
                    test_dataloader,
                    device,
                    max_ans_len=self._resolve_max_ans_len(task_idx),
                    return_predictions=True,
                )

                if self.args.global_rank == 0:
                    safe_task_name = str(task_name).replace("/", "_").replace(":", "_")
                    prediction_file = os.path.join(prediction_root, f"{task_idx}_{safe_task_name}.json")
                    with open(prediction_file, "w", encoding="utf-8") as f:
                        json.dump({"eval": {}, "predictions": prediction_rows}, f, ensure_ascii=False, indent=2)
                    print_rank_0(f"Saved final-test predictions to {prediction_file}", self.args.global_rank)


    def train_one_task(self, task, i_task, epochs):
        # 在单独某个任务上训练
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
        
        #### TRAIN ####
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        loss_eval_dataloader = getattr(self.args, 'loss_eval_task_list', {}).get(
            task,
            eval_dataloader,
        )
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        global_step = 0
        convergence_eval_steps = int(getattr(self.args, 'convergence_eval_steps', 0) or 0)
        engine_start_step = int(getattr(self.model, 'global_steps', 0))
        task_optimizer_step = 0
        last_convergence_step = None
        next_convergence_step = convergence_eval_steps
        local_target_tokens_seen = 0
        training_seconds = 0.0
        latest_train_loss = None

        if convergence_eval_steps > 0:
            self._log_convergence_record(
                task=task,
                event="start",
                optimizer_step=0,
                epoch=0,
                train_loss=None,
                local_target_tokens_seen=0,
                training_seconds=0.0,
                eval_dataloader=loss_eval_dataloader,
                device=device,
            )
            last_convergence_step = 0

        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                global_step += 1
                del batch['sources']
                batch = to_device(batch, device)

                local_target_tokens_seen += int(
                    (batch['labels'][..., 1:] != -100).sum().item()
                )
                train_started = time.perf_counter()
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                latest_train_loss = loss.item()
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                    logging_steps = getattr(self.args, 'logging_steps', 10)
                    if global_step % logging_steps == 0:
                        print_rank_0(f"task={task} epoch={epoch+1} step={global_step} loss={loss.item():.6f}", self.args.global_rank)

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()
                training_seconds += time.perf_counter() - train_started

                engine_global_step = int(
                    getattr(self.model, 'global_steps', engine_start_step + global_step)
                )
                task_optimizer_step = engine_global_step - engine_start_step
                if (
                    convergence_eval_steps > 0
                    and task_optimizer_step >= next_convergence_step
                ):
                    self._log_convergence_record(
                        task=task,
                        event="interval",
                        optimizer_step=task_optimizer_step,
                        epoch=epoch + 1,
                        train_loss=latest_train_loss,
                        local_target_tokens_seen=local_target_tokens_seen,
                        training_seconds=training_seconds,
                        eval_dataloader=loss_eval_dataloader,
                        device=device,
                    )
                    last_convergence_step = task_optimizer_step
                    while next_convergence_step <= task_optimizer_step:
                        next_convergence_step += convergence_eval_steps

            if convergence_eval_steps > 0 and last_convergence_step != task_optimizer_step:
                self._log_convergence_record(
                    task=task,
                    event="epoch_end",
                    optimizer_step=task_optimizer_step,
                    epoch=epoch + 1,
                    train_loss=latest_train_loss,
                    local_target_tokens_seen=local_target_tokens_seen,
                    training_seconds=training_seconds,
                    eval_dataloader=loss_eval_dataloader,
                    device=device,
                )
                last_convergence_step = task_optimizer_step

            # Validate on eval split after each epoch.
            print_rank_0(
                f"***** Evaluating generation metrics, Epoch {epoch+1}/{epochs} on task {task} *****",
                self.args.global_rank)
            eval_result, eval_predictions = self.task_generation_evaluation(
                task,
                eval_dataloader,
                device,
                max_ans_len=self._resolve_max_ans_len(i_task),
                return_predictions=True,
            )
            print_rank_0(f"[task={task}] validation result: {eval_result}", self.args.global_rank)

            self._save_generation_predictions(f"eval-epoch{epoch+1}", i_task, task, eval_result, eval_predictions)

        self.evaluate_seen_tasks_after_training(task, i_task, device)
    
    
    def train_continual(self):
        for i_task, task in enumerate(self.train_task_list):
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            self.save_model(i_task)

    
    def save_model(self, round):
        if self.args.output_dir is not None:
            print_rank_0('saving model to ' + self.args.output_dir + "/" + str(round) + '...', self.args.global_rank)

        if self.args.global_rank == 0:
            save_hf_format(self.model, self.tokenizer, self.args, sub_folder=str(round))

        if self.args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(self.model,
                                  self.args.global_rank,
                                  self.args.output_dir,
                                  zero_stage=self.args.zero_stage,
                                  sub_folder=str(round))
        print_rank_0('Sucessful saving model after round {}'.format(round), self.args.global_rank)
        
