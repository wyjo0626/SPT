# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

import torch
import torch.nn as nn
from tqdm import tqdm

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_BITFIT_BIAS_MODULES_MAPPING,
    ModulesToSaveWrapper,
    # _freeze_adapter,
    _get_submodules,
)

from .config import BitFitConfig
from .layer import BitFitModule


class BitFitModel(BaseTuner):
    """
    Creates Bias-Term Fine-Tuning (BitFit) model from a pretrained transformers model.
    
    The method is described in detail in https://aclanthology.org/2022.acl-short.1
    
    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`BitFitConfig`]): The configuration of the BitFit mopdel.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
    
    Returns:
        `torch.nn.Module`: The BitFit model.
    
    Example:
        ```py
        >>> from transformers import AutoModelForSequenceClassification
        >>> from peft import BitFitModel, BitFitConfig
        
        >>> config = BitFitConfig(
        ...     task_type="SEQ_CLS",
        ...     target_modules=["all"]
        ... )
        
        >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        >>> bitfit_model = BitFitModel(model, config, "default")
        ```
    
    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- the model to be adapted.
        - **peft_config** ([`BitFitConfig`]): The configuration of the BitFit model.
    """
    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name, "bitfit_")
    
    @staticmethod
    def _check_target_module_exists(bitfit_config, key):
        return check_target_module_exists(bitfit_config, key)
    
    def _create_and_replace(self, bitfit_config, adapter_name, target, target_name, parent, **kwargs):
        # If module doesn't have bias, then create bias
        if isinstance(target, BitFitModule):
            target.update_layer(adapter_name)
        else:
            new_module = self._create_new_module(target, adapter_name, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
    
    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "bitfit_base_layer"):
            child = child.bitfit_base_layer
        
        if not hasattr(new_module, "bitfit_base_layer"):
            new_module.weight = child.weight
            
        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "bitfit_base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)
    
    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if not (self.prefix in n and "bias" in n):
                p.requires_grad = False
        
    @staticmethod
    def _create_new_module(target, adapter_name, **kwargs):
        new_module = None
        
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target
        
        new_module = BitFitModule(target, adapter_name, **kwargs)
        
        return new_module
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name) # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)
    
    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config
    
    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            # TODO: Find out what exactly is in Modules ToSaveWrapper
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)
            if isinstance(module, (BitFitModule)):
                if enabled: module.adapter_on()
                else: module.adapter_off()
    
    def enable_adapter_layers(self) -> None:
        """Enable all adapters.
        
        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)
    
    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        self._set_adapter_layers(enabled=False)
    
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_BITFIT_BIAS_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            # If target_modules have None, then include all layers mapping
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_BITFIT_BIAS_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config
