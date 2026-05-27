"""
Inference for SeqSSR-LoRA.

Mirrors infer_anyssr_total.py but:
  - Uses moe_seqssr_lora.py model classes (routing produces ["shared", "task_k"])
  - Loads shared + task-specific adapters from a local checkpoint_dir
  - apply_seqssr_patches() enables PEFT Linear to sum multi-adapter contributions

Effective weight per task t:
    W_eff = W0 + alpha * delta_W_shared + (1 - alpha) * delta_W_task_t

Alpha is baked into each adapter's lora_alpha at training time; at inference we
only need to activate the right adapter pair.

Checkpoint layout (produced by training/main_anamoe.py --CL_method seqssr_lora):
    checkpoint_dir/{t}/shared/      <- shared adapter after training task t
    checkpoint_dir/{t}/task_{t}/    <- task-t-specific adapter
"""

import os
import argparse
import sys
from tqdm import tqdm
from huggingface_hub import hf_hub_download

print('-----------------------------------------------------------------------')

import torch
from torch.utils.data import DataLoader, SequentialSampler
import deepspeed
import json

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
)
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_codetask_dataset, create_executable_dataset, create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model
from utils.code_metrics import bleu as corpus_bleu, smooth_bleu as corpus_smooth_bleu

from training.params import Method2Class, AllDatasetName, AllDatasetNameExecutable

from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.PP import PP, convert_PP_model
from model.Dynamic_network.L2P import convert_L2P_model

from moe_seqssr_lora import (
    NewSdpaAttention, NewLlamaForCausalLM, NewLlamaDecoderLayer, NewLlamaModel,
    NewQwen2SdpaAttention, NewQwen2ForCausalLM, NewQwen2DecoderLayer, NewQwen2Model,
)
from transformers import GenerationConfig
from transformers.models.llama import modeling_llama, LlamaConfig
from transformers.models.qwen2 import modeling_qwen2

from lora_callback import global_callback
from peft import peft_model, PeftModel
import types
from evaluator.compute_metrics import compute_metrics, DATASET_TO_OUTPUT_LANG

# Apply the PEFT patch that enables summed multi-adapter forward (shared + task_k)
import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from model.seqssr_lora import apply_seqssr_patches
apply_seqssr_patches()


def copy_module(module):
    new_module = types.ModuleType(module.__name__ + '_original')
    for attr_name in dir(module):
        if not attr_name.startswith('_'):
            attr_value = getattr(module, attr_name)
            setattr(new_module, attr_name, attr_value)
    return new_module


original_modeling_llama = copy_module(modeling_llama)
modeling_llama.LlamaModel = NewLlamaModel
modeling_llama.LlamaForCausalLM = NewLlamaForCausalLM
modeling_llama.LlamaDecoderLayer = NewLlamaDecoderLayer
modeling_llama.LlamaSdpaAttention = NewSdpaAttention

original_modeling_qwen2 = copy_module(modeling_qwen2)
modeling_qwen2.Qwen2Model = NewQwen2Model
modeling_qwen2.Qwen2ForCausalLM = NewQwen2ForCausalLM
modeling_qwen2.Qwen2DecoderLayer = NewQwen2DecoderLayer
modeling_qwen2.Qwen2SdpaAttention = NewQwen2SdpaAttention


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser(
        description="Inference for SeqSSR-LoRA continual learning method")
    parser.add_argument('--data_path', type=str, default='',
                        help='Path to the training dataset. A single data path.')
    parser.add_argument('--router_weight_path', type=str, default='',
                        help='HF repo ID for FE and router weights.')
    parser.add_argument('--data_output_path', type=str, default='/tmp/data_files/')
    parser.add_argument('--benchmark', type=str,
                        choices=['executable', 'non-executable'], default='non-executable',
                        help='Benchmark to be evaluated: executable or non-executable')
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-Coder-1.5B',
                        required=True,
                        help='Path to pretrained model or model identifier from huggingface.co/models.')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Root output_dir from training (contains 0/, 1/, ... subdirs). '
                             'Shared adapter loaded from checkpoint_dir/{i}/shared/; '
                             'task adapters from checkpoint_dir/{t}/task_{t}/ for t in 0..i.')
    parser.add_argument('--max_prompt_len', type=list_of_strings,
                        default='320,320,256,130,512,256,256,256',
                        help='The maximum prompt length per task.')
    parser.add_argument('--max_ans_len', type=list_of_strings,
                        default='150,256,128,120,300,128,128,128',
                        help='The maximum answer length per task.')
    parser.add_argument('--inference_batch', type=int, default=1,
                        help='Inference batch size.')
    parser.add_argument('--inference_tasks', type=list_of_strings, default='all',
                        help='Datasets to be used.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to store the model.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local_rank for distributed training on gpus')
    parser.add_argument('--inference_output_path', type=str, default=None,
                        help='Where to store inference results.')
    parser.add_argument('--CL_method', default=None, help='continual learning method used')
    parser.add_argument('--do_sample', action='store_true',
                        help='Whether to use sampling for generation.')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--num_return_sequences', type=int, default=5,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on: auto, cpu, cuda, or cuda:<index>.')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def resolve_device(args) -> torch.device:
    if args.device != "auto":
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot see any CUDA device. "
                "Check that the remote environment has an NVIDIA driver, a CUDA-enabled PyTorch build, "
                "and that CUDA_VISIBLE_DEVICES is set correctly."
            )
        return torch.device(args.device)
    if torch.cuda.is_available():
        if args.local_rank is not None and args.local_rank >= 0:
            return torch.device(f"cuda:{args.local_rank}")
        return torch.device("cuda")
    raise RuntimeError(
        "No CUDA device is visible. This inference script is configured to use GPU; "
        "run with --device cpu only if you intentionally want slow CPU inference."
    )


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args)
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] CUDA device count: {torch.cuda.device_count()}")
        print(f"[INFO] CUDA device name: {torch.cuda.get_device_name(device)}")

    if args.inference_tasks[0] == "all":
        if args.benchmark == "non-executable":
            inference_tasks = AllDatasetName
        else:
            inference_tasks = AllDatasetNameExecutable
    else:
        inference_tasks = args.inference_tasks

    task_num = len(inference_tasks)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
        repetition_penalty=args.repetition_penalty,
    )

    def prediction(model, tokenizer, task, test_dataloader, device, generation_config, max_ans_len=None):
        model.eval()
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        moe_ids = []

        if max_ans_len is None:
            max_ans_len = getattr(args, "max_ans_len", 256)

        is_executable = getattr(args, "benchmark", "non-executable") != "non-executable"
        if is_executable:
            num_return_sequences = int(getattr(args, "num_return_sequences", 5))
            top_k = int(getattr(args, "top_k", 0))
            generation_kwargs = generation_config.to_dict()
            generation_kwargs.update({
                "num_return_sequences": num_return_sequences,
                "top_k": top_k,
            })
            generation_config_task = GenerationConfig(**generation_kwargs)
        else:
            num_return_sequences = 1
            generation_config_task = generation_config

        progress_bar = tqdm(total=len(test_dataloader), leave=True, disable=False)
        for step, batch in enumerate(test_dataloader):
            sources_sequences += batch['sources']
            if 'gts' in batch:
                ground_truths += batch['gts']
                del batch['gts']
            elif 'labels' in batch:
                label_tensor = batch['labels']
                for row in label_tensor:
                    valid_ids = row[row != -100].detach().cpu().tolist()
                    gt = tokenizer.decode(valid_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    ground_truths.append(gt)
                del batch['labels']
            else:
                ground_truths += [''] * len(batch['sources'])

            del batch['sources']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]

            with torch.no_grad():
                global_callback.reset()
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = tokenizer.eos_token_id

                generate_ids = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=max_ans_len,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config_task,
                    use_cache=True,
                )

            moe_id = None
            if global_callback.selected_lora_classes:
                moe_entry = global_callback.selected_lora_classes[-1]
                if isinstance(moe_entry, (list, tuple)) and moe_entry:
                    moe_id = moe_entry[0]
                else:
                    moe_id = moe_entry
            elif getattr(model, "model", None) is not None:
                moe_label = getattr(model.model, "label", None)
                if isinstance(moe_label, (list, tuple)) and moe_label:
                    moe_id = moe_label[0]
                else:
                    moe_id = moe_label

            print(f"Predicted MoE ID: {moe_id}")
            sequences = tokenizer.batch_decode(
                generate_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if is_executable and num_return_sequences > 1:
                batch_preds = [
                    sequences[i:i + num_return_sequences]
                    for i in range(0, len(sequences), num_return_sequences)
                ]
                predicted_sequences.extend(batch_preds)
                moe_ids.extend([moe_id] * len(batch_preds))
            else:
                predicted_sequences += sequences
                moe_ids.extend([moe_id] * len(sequences))

            progress_bar.update(1)
            progress_bar.set_description(f"Test step {step}", refresh=False)

        return sources_sequences, predicted_sequences, ground_truths, moe_ids

    def _task_eval_from_predictions(task, sources_sequences, predicted_sequences, ground_truths):
        calc_codebleu = task not in ('CodeSearchNet', 'TheVault_Csharp')
        return compute_metrics(predicted_sequences, ground_truths,
                               calc_codebleu=calc_codebleu,
                               language=DATASET_TO_OUTPUT_LANG.get(task, None))

    def save_inference_results(evaluation_result, sources_sequences, predicted_sequences,
                               ground_truths, moe_ids, i_task, task):
        df = {"eval": evaluation_result}
        os.makedirs(args.inference_output_path, exist_ok=True)
        if len(moe_ids) != len(predicted_sequences):
            moe_ids_padded = (moe_ids + [None] * len(predicted_sequences))[:len(predicted_sequences)]
        else:
            moe_ids_padded = moe_ids
        prediction_rows = [
            {
                "source": source,
                "ground-truth": gt,
                "prediction": pred,
                "moe_id": moe_id,
            }
            for source, gt, pred, moe_id in zip(sources_sequences, ground_truths, predicted_sequences, moe_ids_padded)
        ]
        df["predictions"] = prediction_rows
        output_file = os.path.join(args.inference_output_path, f"results-{i_task}-{task}.json")
        with open(output_file, "w", encoding='utf-8') as file:
            json.dump(df, file, ensure_ascii=False)
            file.write("\n")
        print(f"[INFO] Saved inference results to {output_file}", flush=True)

    # Infer at each step i (mirrors anyssr: loop over the last step only, or all steps)
    for i in range(task_num - 1, task_num):
        tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
        model_dtype = torch.float16 if device.type == "cuda" else torch.float32

        if "llama" in args.model_name_or_path.lower():
            model = modeling_llama.LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                tasks=i + 1,
                torch_dtype=model_dtype,
            )
        elif "qwen" in args.model_name_or_path.lower():
            model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(
                args.model_name_or_path,
                tasks=i + 1,
                torch_dtype=model_dtype,
            )

        # Load FE and router weights from HF (same source as Any-SSR)
        fe_path = hf_hub_download(
            repo_id=args.router_weight_path,
            filename=f"step{i}_fe_weight.pth",
            repo_type="model",
        )
        router_path = hf_hub_download(
            repo_id=args.router_weight_path,
            filename=f"step{i}_router_weight.pth",
            repo_type="model",
        )
        fe_weight = torch.load(fe_path, map_location="cpu").to(model_dtype)
        classifier_weight = torch.load(router_path, map_location="cpu").transpose(0, 1).to(model_dtype)

        # Load shared adapter from the final step's checkpoint.
        # Use load_adapter (in-place) so model stays as NewQwen2ForCausalLM and
        # model.model still points to NewQwen2Model (which holds fe / moe_classifier).
        shared_ckpt = os.path.join(args.checkpoint_dir, str(i), "shared")
        model.load_adapter(shared_ckpt, adapter_name="shared")
        print(f"[INFO] Loaded shared adapter from {shared_ckpt}", flush=True)

        # Load task-specific adapters for tasks 0..i
        for t in range(i + 1):
            task_ckpt = os.path.join(args.checkpoint_dir, str(t), f"task_{t}")
            model.load_adapter(task_ckpt, adapter_name=f"task_{t}")
            print(f"[INFO] Loaded task_{t} adapter from {task_ckpt}", flush=True)

        print("Successfully loaded adapters:", list(model.peft_config.keys()))
        lora_params = [name for name, _ in model.named_parameters() if "lora" in name]
        print(f"Total LoRA tensors found in memory: {len(lora_params)}")
        if lora_params:
            print("Sample LoRA layer path:", lora_params[0])

        model.model.moe_classifier.weight = torch.nn.Parameter(classifier_weight)
        model.model.fe.weight = torch.nn.Parameter(fe_weight)
        model.to(device)

        cur_inference_tasks = inference_tasks[0:i + 1]
        for inference_task_id in range(len(cur_inference_tasks)):
            inference_task = inference_tasks[inference_task_id]

            # Prepare the data
            if args.benchmark == "non-executable":
                train, test, infer_dataset = create_codetask_dataset(
                    inference_task, args.seed, -1, -1, -1)
            else:
                train, test, infer_dataset = create_executable_dataset(
                    inference_task, args.seed, -1, -1, -1)

            inf_data_collator = DataCollator(
                tokenizer,
                model=model,
                padding="longest",
                max_prompt_len=int(args.max_prompt_len[inference_task_id]),
                max_ans_len=int(args.max_ans_len[inference_task_id]),
                pad_to_multiple_of=8,
                inference=True,
            )
            infer_dataloader = DataLoader(
                infer_dataset,
                collate_fn=inf_data_collator,
                sampler=SequentialSampler(infer_dataset),
                batch_size=args.inference_batch,
            )

            assert tokenizer.padding_side == 'left'
            assert tokenizer.truncation_side == "left"

            print(f"***** Start inference of step {i}: task {inference_task} *****")
            sources_sequences, predicted_sequences, ground_truths, moe_ids = prediction(
                model, tokenizer, inference_task, infer_dataloader, device,
                generation_config,
                max_ans_len=int(args.max_ans_len[inference_task_id]),
            )

            if args.benchmark == "non-executable":
                evaluation_result = _task_eval_from_predictions(
                    inference_task, sources_sequences, predicted_sequences, ground_truths)
            else:
                evaluation_result = {}

            print("***** Saving inference results *****")
            save_inference_results(
                evaluation_result, sources_sequences, predicted_sequences,
                ground_truths, moe_ids, inference_task_id, inference_task,
            )


if __name__ == "__main__":
    main()
