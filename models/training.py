from typing import Optional
from copy import deepcopy
import os
import logging
import torch
import numpy as np

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
    CONFIG_NAME,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    ADAPTER_CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
    is_accelerate_available,
    is_safetensors_available,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations import deepspeed_load_checkpoint
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from peft import PeftModel, XPromptTuningConfig, RPromptTuningConfig, PromptLearningConfig
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


def sim(matrix1, matrix2):
    cosine_similarity = np.dot(matrix1, matrix2) / (np.linalg.norm(matrix1) * np.linalg.norm(matrix2))
    return cosine_similarity


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

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )

        if is_fsdp_ckpt and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_ckpt
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
            weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if hasattr(self.args, "fp16") and self.args.fp16 is True:
                        logger.warning(
                            "Enabling FP16 and loading from smp < 1.10 checkpoint together is not suppported."
                        )
                    state_dict = torch.load(
                        weights_file,
                        map_location="cpu",
                        **weights_only_kwarg,
                    )
                    # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
                    # release memory
                    del state_dict
            elif self.is_fsdp_enabled:
                load_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    resume_from_checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                else:
                    state_dict = torch.load(
                        weights_file,
                        map_location="cpu",
                        **weights_only_kwarg,
                    )

                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, False)
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        elif isinstance(model, PeftModel):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                if os.path.exists(resume_from_checkpoint):
                    model.load_adapter(resume_from_checkpoint, model.active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)


class BaseSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(TrainCallback(self))
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, ProgressCallback):
                self.remove_callback(callback)
        self.add_callback(TrainProgressCallback(self))
        self.cosine_similarity = None
    
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

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )

        if is_fsdp_ckpt and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_ckpt
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
            weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if hasattr(self.args, "fp16") and self.args.fp16 is True:
                        logger.warning(
                            "Enabling FP16 and loading from smp < 1.10 checkpoint together is not suppported."
                        )
                    state_dict = torch.load(
                        weights_file,
                        map_location="cpu",
                        **weights_only_kwarg,
                    )
                    # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
                    # release memory
                    del state_dict
            elif self.is_fsdp_enabled:
                load_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    resume_from_checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                else:
                    state_dict = torch.load(
                        weights_file,
                        map_location="cpu",
                        **weights_only_kwarg,
                    )

                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, False)
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        elif isinstance(model, PeftModel):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                if os.path.exists(resume_from_checkpoint):
                    model.load_adapter(resume_from_checkpoint, model.active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.args.cosine_similarity:
            inputs["output_hidden_states"] = True

        outputs = model(**inputs)
        
        if self.args.cosine_similarity and isinstance(model, PeftModel) and isinstance(model.active_peft_config, PromptLearningConfig):
            sim_ = []
            for i in range(model.active_peft_config.num_virtual_tokens):
                sim_.append(
                    sim(
                    outputs["encoder_hidden_states"][0][0][i].detach().cpu(),
                    outputs["encoder_hidden_states"][12][0][i].detach().cpu(),
                    )
                )
            self.cosine_similarity = float(np.mean(sim_))
        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        return (loss, outputs) if return_outputs else loss


class TrainCallback(TrainerCallback):
    def  __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
        self.is_eval_training = False
        
        if (isinstance(trainer.model, PeftModel) and 
            (isinstance(trainer.model.active_peft_config, XPromptTuningConfig) or isinstance(trainer.model.active_peft_config, RPromptTuningConfig))
            ):
            self.prune = trainer.model.prompt_encoder[trainer.model.active_adapter]
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate and self._trainer.args.eval_training:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            self.is_eval_training = True
            return control_copy

        if control.should_evaluate:
            return

        if hasattr(self, "prune"):
            self.prune.estimate_token_importance(self._trainer, state.global_step)
            self.prune.estimate_piece_importance(self._trainer, state.global_step)
    
    def on_step_end(self, args, state, control, **kwargs):
        if self._trainer.args.cosine_similarity and self._trainer.cosine_similarity is not None:
            self._trainer.log({"cosine_similarity": self._trainer.cosine_similarity})
        
        if control.should_evaluate and self._trainer.args.eval_training:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            self.is_eval_training = True
            return control_copy

        if control.should_evaluate:
            return
        
        if hasattr(self, "prune"):
            self.prune.estimate_token_importance(self._trainer, state.global_step)
            self.prune.estimate_piece_importance(self._trainer, state.global_step)
    
    def on_evaluate(self, args, state, control, **kwargs):
        if self.is_eval_training:
            self.is_eval_training = False
            return
        
        if not self.is_eval_training and hasattr(self, "prune"):
            self.prune.estimate_token_importance(self._trainer, state.global_step)
            self.prune.estimate_piece_importance(self._trainer, state.global_step)

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
