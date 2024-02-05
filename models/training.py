from typing import Optional
from copy import deepcopy
import os
import logging
import torch

from transformers import (
    Trainer,
    Seq2SeqTrainer,
    TrainerCallback,
    ProgressCallback,
    PreTrainedModel,
)
from transformers.modeling_utils import (
    unwrap_model,
    load_sharded_checkpoint,
)
from transformers.utils import (
    WEIGHTS_NAME,
    SAFE_WEIGHTS_NAME,
    ADAPTER_CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
    is_accelerate_available,
    is_safetensors_available,
)
from transformers.integrations import deepspeed_load_checkpoint
from peft import PeftModel, XPromptTuningConfig
from utils.general import colorstr, colorformat, emojis

logger = logging.getLogger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

if is_safetensors_available():
    import safetensors.torch

class BaseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(TrainCallback(self))
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, ProgressCallback):
                self.remove_callback(callback)
        self.add_callback(TrainProgressCallback(self))
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    
    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, self.state.best_model_checkpoint)
        elif self.is_fsdp_enabled:
            load_result = load_fsdp_model(
                self.accelerator.state.fsdp_plugin, self.accelerator, model, self.state.best_model_checkpoint
            )
        elif (
            os.path.exists(best_model_path)
            or os.path.exists(best_safe_model_path)
            or os.path.exists(best_adapter_model_path)
            or os.path.exists(best_safe_adapter_model_path)
        ):
            has_been_loaded = True
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(self.state.best_model_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=self.state.best_model_checkpoint,
                        tag=WEIGHTS_NAME,
                        partial=False,
                        load_optimizer=False,
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                        state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                    else:
                        state_dict = torch.load(best_model_path, map_location="cpu", weights_only=True)

                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
            else:
                if isinstance(model, PeftModel):
                    # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
                    if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                        if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                            model.load_adapter(self.state.best_model_checkpoint, model.active_adapter)
                            # Load_adapter has no return value present, modify it when appropriate.
                            from torch.nn.modules.module import _IncompatibleKeys

                            load_result = _IncompatibleKeys([], [])
                        else:
                            logger.warning(
                                "The intermediate checkpoints of PEFT may not be saved correctly, "
                                f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                                "Check some examples here: https://github.com/huggingface/peft/issues/96"
                            )
                            has_been_loaded = False
                    else:
                        logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
                        has_been_loaded = False
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                        state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                    else:
                        state_dict = torch.load(best_model_path, map_location="cpu", weights_only=True)

                    # If the model is on the GPU, it still works!
                    # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                    # which takes *args instead of **kwargs
                    load_result = model.load_state_dict(state_dict, False)
                if not is_sagemaker_mp_enabled() and has_been_loaded:
                    self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)):
            load_result = load_sharded_checkpoint(
                model, self.state.best_model_checkpoint, strict=is_sagemaker_mp_enabled()
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )


class BaseSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(TrainCallback(self))
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, ProgressCallback):
                self.remove_callback(callback)
        self.add_callback(TrainProgressCallback(self))
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    
    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, self.state.best_model_checkpoint)
        elif self.is_fsdp_enabled:
            load_result = load_fsdp_model(
                self.accelerator.state.fsdp_plugin, self.accelerator, model, self.state.best_model_checkpoint
            )
        elif (
            os.path.exists(best_model_path)
            or os.path.exists(best_safe_model_path)
            or os.path.exists(best_adapter_model_path)
            or os.path.exists(best_safe_adapter_model_path)
        ):
            has_been_loaded = True
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(self.state.best_model_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=self.state.best_model_checkpoint,
                        tag=WEIGHTS_NAME,
                        partial=False,
                        load_optimizer=False,
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                        state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                    else:
                        state_dict = torch.load(best_model_path, map_location="cpu", weights_only=True)

                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
            else:
                if isinstance(model, PeftModel):
                    # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
                    if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                        if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                            model.load_adapter(self.state.best_model_checkpoint, model.active_adapter)
                            # Load_adapter has no return value present, modify it when appropriate.
                            from torch.nn.modules.module import _IncompatibleKeys

                            load_result = _IncompatibleKeys([], [])
                        else:
                            logger.warning(
                                "The intermediate checkpoints of PEFT may not be saved correctly, "
                                f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                                "Check some examples here: https://github.com/huggingface/peft/issues/96"
                            )
                            has_been_loaded = False
                    else:
                        logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
                        has_been_loaded = False
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                        state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                    else:
                        state_dict = torch.load(best_model_path, map_location="cpu", weights_only=True)

                    # If the model is on the GPU, it still works!
                    # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                    # which takes *args instead of **kwargs
                    load_result = model.load_state_dict(state_dict, False)
                if not is_sagemaker_mp_enabled() and has_been_loaded:
                    self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)):
            load_result = load_sharded_checkpoint(
                model, self.state.best_model_checkpoint, strict=is_sagemaker_mp_enabled()
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )


class TrainCallback(TrainerCallback):
    def  __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
        self.is_xprompt = False
        self.is_training_eval = False
        
        if (isinstance(trainer.model, PeftModel) and 
            isinstance(trainer.model.active_peft_config, XPromptTuningConfig)):
            self.is_xprompt = True
            self.xprompt = trainer.model.prompt_encoder[trainer.model.active_adapter]
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            self.is_training_eval = True
            return control_copy
    
    def on_step_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            self.is_training_eval = True
            return control_copy
        
        if self.is_xprompt:
            self.xprompt.estimate_token_importance(self._trainer, args.max_steps, state.global_step)
    
    def on_evaluate(self, args, state, control, **kwargs):
        if self.is_training_eval:
            self.is_training_eval = False
            return
        
        if not self.is_training_eval and self.is_xprompt:
            self.xprompt.estimate_token_importance(self._trainer, args.max_steps, state.global_step)

    def on_predict(self, args, state, control, metrics, **kwargs):
        # Save predict result.
        self._trainer.log(metrics)


class TrainProgressCallback(ProgressCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log the metrics.
        # Best metrics may be logged on color string to check best condition.
        items = []
        best = state.best_metric if state.best_metric is not None else 0
        
        for key, value in logs.items():
            if key == "eval_" + args.metric_for_best_model and best < value:
                items.append(f"'{colorstr('bright_cyan', 'bold', key)}': {colorstr('bright_cyan', 'bold', str(value))}")
            else:
                items.append(f"'{key}': {value}")
        
        if state.is_world_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flops", None)
            self.training_bar.write('{' + ', '.join(items) + '}')
