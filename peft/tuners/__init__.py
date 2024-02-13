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
from .adaption_prompt import AdaptionPromptConfig, AdaptionPromptModel
from .lora import LoraConfig, LoraModel, LoftQConfig
from .adalora import AdaLoraConfig, AdaLoraModel
from .p_tuning import PromptEncoder, PromptEncoderConfig, PromptEncoderReparameterizationType
from .prefix_tuning import PrefixEncoder, PrefixTuningConfig
from .prompt_tuning import PromptEmbedding, PromptTuningConfig, PromptTuningInit
from .residual_prompt_tuning import ResidualPromptTuningConfig, ResidualPromptTuningInit, ResidualPromptTuningReparameterizationType, ResidualPromptEmbedding, ResidualMLP
from .bitfit import BitFitConfig, BitFitModel
from .xprompt_tuning import XPromptEmbedding, XPromptTuningConfig, XPromptTuningInit