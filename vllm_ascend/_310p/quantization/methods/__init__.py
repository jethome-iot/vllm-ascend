#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
# [310P] Registers a W8A16 linear scheme for Ascend 310P.
#
# ROOT CAUSE / FIX (verified on Atlas 300I Duo, TP=4):
# ``npu_weight_quant_batchmatmul`` (aclnnWeightQuantBatchMatmulV2) on 310P needs the weight in
# plain ND ``[K, N]`` layout. The base ``w8a16.py`` applies ``maybe_trans_nz`` (FRACTAL_NZ),
# which on 310P yields "task not supported" (EE9999 / 161002) in the real forward pipeline
# (an isolated micro-repro with NZ passes, but the full pipeline does not). The W8A8 kernel
# ``npu_quant_matmul`` is the opposite and needs NZ. So here: transpose to ``[K, N]`` +
# contiguous, no NZ cast; flatten the scale/offset to ``[N]``. ``apply()`` is inherited
# unchanged from the base method.
#
import torch

from vllm_ascend.quantization.methods.w8a16 import AscendW8A16LinearMethod

from . import (
    w8a8_dynamic,  # noqa: F401
    w8a8_static,  # noqa: F401
    w8a8s,  # noqa: F401
    w8a8sc,  # noqa: F401
)
from .registry import register_scheme


@register_scheme("W8A16", "linear")
class AscendW8A16LinearMethod310(AscendW8A16LinearMethod):
    """310P W8A16 linear: ND [K, N] weight (no FRACTAL_NZ), flat [N] scale/offset."""

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
