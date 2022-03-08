# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional, Sequence

import torch


def _get_autocast_dtypes() -> Sequence[torch.dtype]:
    if torch.cuda.is_bf16_supported():
        return [torch.half, torch.bfloat16]
    return [torch.half]


def _get_current_dtype(dtype: Optional[torch.dtype] = None) -> torch.dtype:
    if not torch.is_autocast_enabled():
        return torch.float or dtype
    else:
        return torch.get_autocast_gpu_dtype()


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())
