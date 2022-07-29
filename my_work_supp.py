############三角乘法更新依赖的函数，供my_work.py调用
import torch.nn as nn
import torch
from typing import Optional, Callable, List, Tuple, Sequence
class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        # d = x.dtype
        # if (d is torch.bfloat16 and not deepspeed.utils.is_initialized()):
        #     with torch.cuda.amp.autocast(enabled=False):
        #         out = nn.functional.layer_norm(
        #             x,
        #             self.c_in,
        #             self.weight.to(dtype=d),
        #             self.bias.to(dtype=d),
        #             self.eps
        #         )
        # else:
        out = nn.functional.layer_norm(
            x,
            self.c_in,
            self.weight,
            self.bias,
            self.eps,
        )
        return out
def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    # c = first_inds
    # for i in inds:
    #     c=c+ [zero_index + i ]
    # d=tensor.permute(c)
    return tensor.permute(first_inds + [zero_index + i for i in inds])
#
# a=torch.rand(1,3,3,2)
# b=permute_final_dims(a,(2,0,1))
# e=a.permute([0,-1,-2,-3])
# f=a.permute([0,3,1,2])
# g=f=a.permute([0,3,1,2])
# print(a)
# print(b)
# # print(e)
# # print(f)
# print(g)
