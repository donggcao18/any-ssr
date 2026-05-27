"""SeqSSR-LoRA continual learning method.

Effective weight update per task t:
    W_eff = W0 + alpha * delta_W_shared + (1 - alpha) * delta_W_task_t

- delta_W_shared  : one shared LoRA adapter, trained sequentially across all tasks
- delta_W_task_t  : task-specific LoRA adapter, freshly created and frozen after task t
- alpha           : fixed scalar in [0, 1] passed via --alpha

Adapter structure mirrors anamoe (start_layer=4, target q_proj + v_proj on layers 4-27).

Checkpoint layout after training task t:
    output_dir/{t}/shared/          <- shared adapter weights
    output_dir/{t}/task_{t}/        <- task-t-specific adapter weights
    output_dir/{t}/tokenizer.json   <- tokenizer

Resume: set --start_task_id to the next task to train.
"""

import os
import torch
from tqdm import tqdm

from model.lora import lora
from utils.utils import print_rank_0, to_device


# ---------------------------------------------------------------------------
# PEFT 0.6.2 compatibility patch
# ---------------------------------------------------------------------------

_SEQSSR_PATCHED = False


def apply_seqssr_patches():
    """
    Monkey-patch PEFT 0.6.2's Linear.forward so that active_adapter may be a
    list of adapter names.  When a list is given the contributions of all
    listed adapters are summed (alpha weighting is baked into each adapter's
    lora_alpha at creation time).

    Also patches LoraModel.set_adapter so it accepts a list.

    Idempotent: safe to call multiple times.
    """
    global _SEQSSR_PATCHED
    if _SEQSSR_PATCHED:
        return

    import torch.nn.functional as F
    import peft.tuners.lora as _lora_mod

    Linear = _lora_mod.Linear
    LoraModel = _lora_mod.LoraModel

    # --- patch Linear.forward ---
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        # active_adapter may be a str or list[str]; also fall back to
        # _active_adapter if the property is shadowed
        active = getattr(self, "_active_adapter", None)
        if active is None:
            active = getattr(self, "active_adapter", "default")
        if isinstance(active, str):
            active = [active]

        # base projection (no LoRA)
        w = self.weight.T if self.fan_in_fan_out else self.weight
        base = F.linear(x, w, bias=self.bias)

        disable = getattr(self, "_disable_adapters",
                          getattr(self, "disable_adapters", False))
        merged = getattr(self, "merged", False)

        if disable or merged:
            return base.to(previous_dtype)

        result = base
        for adapter in active:
            if adapter not in self.lora_A or self.r.get(adapter, 0) <= 0:
                continue
            x_cast = x.to(self.lora_A[adapter].weight.dtype)
            result = result + (
                self.lora_B[adapter](
                    self.lora_A[adapter](self.lora_dropout[adapter](x_cast))
                )
                * self.scaling[adapter]
            )
        return result.to(previous_dtype)

    Linear.forward = _forward

    # --- patch LoraModel.set_adapter to accept list ---
    import warnings

    def _set_adapter(self, adapter_name):
        LoraLayer = _lora_mod.LoraLayer
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn(
                        "Adapter cannot be set when model is merged. Unmerging first."
                    )
                    module.unmerge()
                # Use _active_adapter directly to avoid read-only-property guards
                # introduced in newer PEFT refactors that may coexist in env
                try:
                    module.active_adapter = adapter_name
                except AttributeError:
                    object.__setattr__(module, "_active_adapter", adapter_name)
        self.active_adapter = adapter_name

    LoraModel.set_adapter = _set_adapter

    _SEQSSR_PATCHED = True


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _activate(peft_model, adapter_names):
    """Set active_adapter on all LoRA layers to adapter_names (str or list)."""
    peft_model.base_model.set_adapter(adapter_names)


# ---------------------------------------------------------------------------
# SeqSSRLoRA class
# ---------------------------------------------------------------------------


class SeqSSRLoRA(lora):
    """Shared-sequential + task-specific LoRA with fixed alpha blending."""

    def __init__(self, model, tokenizer, optimizer,
                 train_task_list, eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer,
                         train_task_list, eval_task_list, test_task_list, args)
        self.alpha = float(getattr(args, "alpha", 0.5))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _peft(self):
        """Return the PeftModel wrapped by the DeepSpeed engine."""
        return self.model.module

    def _switch_to(self, task_id):
        """Activate [shared, task_{task_id}] adapters for forward pass."""
        _activate(self._peft(), ["shared", f"task_{task_id}"])

    # ------------------------------------------------------------------
    # Main training loop (one task per invocation)
    # ------------------------------------------------------------------

    def train_continual(self):
        """Train exactly the task at start_task_id, then save.

        Each shell invocation handles one task (identical pattern to SeqLoRA).
        The training script is responsible for incrementing --start_task_id.
        """
        start_task_id = int(getattr(self.args, "start_task_id", 0))
        task_items = list(self.train_task_list.items())

        if start_task_id >= len(task_items):
            print_rank_0(
                f"start_task_id={start_task_id} >= number of tasks "
                f"({len(task_items)}); nothing to train.",
                self.args.global_rank,
            )
            return

        i_task = start_task_id
        task, _ = task_items[i_task]

        self._switch_to(i_task)
        self._train_seqssr(task, i_task, int(self.args.num_train_epochs[i_task]))
        self.save_model(i_task)

    # ------------------------------------------------------------------
    # Per-task training (mirrors base_model.train_one_task but switches
    # adapters when evaluating seen tasks)
    # ------------------------------------------------------------------

    def _train_seqssr(self, task, i_task, epochs):
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            device = torch.device("cuda", self.args.local_rank)

        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(
            total=total_steps, leave=True,
            disable=(self.args.global_rank != 0)
        )
        global_step = 0

        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, "
                f"Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank,
            )
            # Restore training adapters at the top of each epoch
            self._switch_to(i_task)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                global_step += 1
                del batch["sources"]
                batch.pop("indices", None)
                batch = to_device(batch, device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss

                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}",
                        refresh=False,
                    )
                    if global_step % getattr(self.args, "logging_steps", 10) == 0:
                        print_rank_0(
                            f"task={task} epoch={epoch+1} "
                            f"step={global_step} loss={loss.item():.6f}",
                            self.args.global_rank,
                        )

                self.model.backward(loss)
                self.model.step()

            # Per-epoch evaluation on current task
            print_rank_0(
                f"***** Evaluating, Epoch {epoch+1}/{epochs}, task={task} *****",
                self.args.global_rank,
            )
            eval_result, eval_preds = self.task_generation_evaluation(
                task, eval_dataloader, device,
                max_ans_len=self._resolve_max_ans_len(i_task),
                return_predictions=True,
            )
            print_rank_0(
                f"[task={task}] val: {eval_result}", self.args.global_rank
            )
            self._save_generation_predictions(
                f"eval-epoch{epoch+1}", i_task, task, eval_result, eval_preds
            )

        # After all training epochs: test on all tasks seen so far,
        # switching to the appropriate adapter pair for each.
        for seen_idx, (test_task, test_ds) in enumerate(
            list(self.test_task_list.items())[: i_task + 1]
        ):
            self._switch_to(seen_idx)
            self.model.eval()
            print_rank_0(
                f"***** Testing on {test_task} after training {task} *****",
                self.args.global_rank,
            )
            test_result, test_preds = self.task_generation_evaluation(
                test_task, test_ds, device,
                max_ans_len=self._resolve_max_ans_len(seen_idx),
                return_predictions=True,
            )
            print_rank_0(
                f"[task={test_task}] post-train test: {test_result}",
                self.args.global_rank,
            )
            self._save_generation_predictions(
                "test-after-task", i_task, test_task, test_result, test_preds
            )

        # Restore current task adapters
        self._switch_to(i_task)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_model(self, i_task):
        if self.args.output_dir is None:
            return
        print_rank_0(
            f"Saving SeqSSR-LoRA adapters for task {i_task}...",
            self.args.global_rank,
        )
        if self.args.global_rank == 0:
            out_dir = os.path.join(self.args.output_dir, str(i_task))
            os.makedirs(out_dir, exist_ok=True)
            # save_pretrained with selected_adapters writes each adapter to
            # out_dir/<adapter_name>/ (adapter_config.json + adapter_model.bin)
            self._peft().save_pretrained(
                out_dir,
                selected_adapters=["shared", f"task_{i_task}"],
            )
            self.tokenizer.save_pretrained(out_dir)
            print_rank_0(
                f"Saved adapters to {out_dir}", self.args.global_rank
            )

    # ------------------------------------------------------------------
    # Final inference (after full CL sequence)
    # ------------------------------------------------------------------

    def test_all_tasks_and_save_predictions(self):
        """Override to switch to the correct task adapter per task."""
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            device = torch.device("cuda", self.args.local_rank)

        for task_idx, (task_name, test_dataloader) in enumerate(
            self.test_task_list.items()
        ):
            self._switch_to(task_idx)
            self.model.eval()
            print_rank_0(
                f"***** Final testing on task {task_name} *****",
                self.args.global_rank,
            )
            test_result, prediction_rows = self.task_generation_evaluation(
                task_name, test_dataloader, device,
                max_ans_len=self._resolve_max_ans_len(task_idx),
                return_predictions=True,
            )
            self._save_generation_predictions(
                "final-test", task_idx, task_name, test_result, prediction_rows
            )
