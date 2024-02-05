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
import copy
import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer


class BitFitLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter parameter (bias)
    adapter_layer_names = ("bias_adapters", )
    
    # Save and Fix the original bias
    _original_bias = "original_bias"
    
    def __init__(self, base_layer: nn.Module, **kwargs):
        self.bitfit_base_layer = base_layer
        self._disable_adapters = False
        self.kwargs = kwargs
        # For bias parameter
        self.bias_adapters = nn.ParameterDict({})
        
        base_layer = self.get_base_layer()
        # TODO: we have to implement the layer that don't have register_parameter method 
        # such as T5LayerNorm. Later, Implement the customization, then also complete adapter_on/off
        if isinstance(base_layer, nn.Linear):
            out_features = base_layer.out_features
        elif isinstance(base_layer, nn.LayerNorm):
            out_features = base_layer.normalized_shape[0]
        elif isinstance(base_layer, nn.Conv2d) or isinstance(base_layer, nn.Conv1d):
            out_features = base_layer.out_channels
        elif isinstance(base_layer, Conv1D):
            _, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif isinstance(base_layer, nn.Embedding):
            out_features = base_layer.embedding_dim
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        
        if hasattr(base_layer, "bias"):
            if base_layer.bias is not None:
                self.bias_adapters[self._original_bias] = base_layer.bias
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        
        self.out_features = out_features

    def get_base_layer(self) -> nn.Module:
        """
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
        base_layer = self
        while hasattr(base_layer, "bitfit_base_layer"):
            base_layer = base_layer.bitfit_base_layer
        return base_layer

    def update_layer(self, adapter_name):
        # Add trainable parameter as bias
        if self.original_bias is not None:
            self.bias_adapters[adapter_name] = copy.deepcopy(self.original_bias)
        else:
            self.bias_adapters[adapter_name] = nn.Parameter(torch.zeros(self.out_features))
        
        weight = getattr(self.bitfit_base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(self.weight.device)
        
        self.set_adapter(self.active_adapter)

    # Adapter (bias) on base layer
    def adapter_on(self, adapter_name: str = "default"):
        if self.has_register:
            self.bitfit_base_layer.register_parameter("bias", self.bias_adapters[adapter_name])
        else:
            raise ValueError(f"Unsupported layer type {type(bitfit_base_layer)}")

    # Adapter (bias) off base layer. replace adapter to original bias
    def adapter_off(self):
        if self.has_register:
            self.bitfit_base_layer.register_parameter("bias", self.original_bias)
        else:
            raise ValueError(f"Unsupported layer type {type(bitfit_base_layer)}")

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Delete an adapter from the layer
        
        Maintain the original bias

        Args:
            adapter_name (`str`): The name of the adapter to delete
        """
        if adapter_name == _original_bias:
            warnings.warn(
                f"Adapter {adapter_name} is not deleted because it it the original bias. "
            )
            return
        
        super().delete_adapter(adapter_name)

    @property
    def has_register(self) -> bool:
        # if layer has register_parameter
        # TODO: if layer is customized module, or others such as T5LayerNorm, can't add bias.
        # later implement this method to other function.
        return hasattr(self.bitfit_base_layer, "register_parameter")
    
    @property
    def original_bias(self):
        # use a property to ensure that original bias is not set directly, fix the original bias
        return self.bias_adapters.get(self._original_bias)


class BitFitModule(nn.Module, BitFitLayer):
    # BitFit implemented in a dense layer
    def __init__(self, base_layer, adapter_name: str, **kwargs):
        super().__init__()
        BitFitLayer.__init__(self, base_layer=base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name)
    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.bitfit_base_layer(x, *args, **kwargs)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "bitfit." + rep
