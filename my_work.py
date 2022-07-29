############三角乘法更新：下面的tri_model
from functools import partialmethod
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import Linear
from my_work_supp import LayerNorm,permute_final_dims
from dropout import DropoutRowwise

class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """
    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        # self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_a_g = Linear(self.c_z, self.c_hidden)
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        # self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_g = Linear(self.c_z, self.c_hidden)
        # self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z)
        # self.linear_z = Linear(self.c_hidden, self.c_z, init="final")
        self.linear_z = Linear(self.c_hidden, self.c_z)
        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("This method needs to be overridden")

    def forward(self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])#1,400,400的全1矩阵

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)    #归一化，1，400，400，128
        a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z))        #1,400,400,64(我设的隐藏层)
        a = a * mask
        b = self.linear_b_p(z) * self.sigmoid(self.linear_b_g(z))
        b = b * mask
        x = self._combine_projections(a, b)
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        z = x * g

        return z


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """
    def _combine_projections(self,
        a: torch.Tensor,  # [*, N_i, N_k, C]
        b: torch.Tensor,  # [*, N_j, N_k, C]
    ):
        # [*, C, N_i, N_j]
        p = torch.matmul(
            permute_final_dims(a, (2, 0, 1)),            #就是交换了[*, N_i, N_k, C]中N_i和C的位置[*, C, N_i, N_k]
            permute_final_dims(b, (2, 1, 0)),            #[*, C, N_k,N_j]
        )                                                 #[N_i, N_k]*[N_k,N_j]


        # [*, N_i, N_j, C]
        return permute_final_dims(p, (1, 2, 0))


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """
    def _combine_projections(self,
        a: torch.Tensor,  # [*, N_k, N_i, C]
        b: torch.Tensor,  # [*, N_k, N_j, C]
    ):
        # [*, C, N_i, N_j]
        # aa=permute_final_dims(a, (2, 1, 0))            #a是1，400，400，64，aa是1，64，400，400，
        # aa
        #两个乘法更新的区别在这里，
        p = torch.matmul(
            permute_final_dims(a, (2, 1, 0)),            #[*,C,N_i, N_k]
            permute_final_dims(b, (2, 0, 1)),            #[*,C,N_k, N_j]
        )

        # [*, N_i, N_j, C]
        return permute_final_dims(p, (1, 2, 0))

class tri_model(nn.Module):
    def __init__(self,c_z,
            c_hidden_mul,):
        super(tri_model, self).__init__()
        # self.P=FullyConnectedEmbed()
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.pair_dropout=0.3   #
        self.ps_dropout_row_layer = DropoutRowwise(self.pair_dropout)
    def forward(self, z):
        a=self.tri_mul_in(z, mask=None)       #1,400,400,128
        # a=self.ps_dropout_row_layer(a)
        # a
        z = z + self.ps_dropout_row_layer(self.tri_mul_in(z, mask=None))
        z = z + self.ps_dropout_row_layer(self.tri_mul_out(z, mask=None))
        return z

# c_z = 128
# c_hidden_mul = 64
# z=torch.rand(1,400,400,c_z)
# model=model_model(c_z,c_hidden_mul)
# zz=model(z)
# total = sum([param.nelement() for param in model.parameters()])
# print("Number of parameter: %.2fM" % (total / 1e6))
# distance=torch.rand(1,4,4)
# import torch
# one = torch.ones_like(distance)
# distance = torch.where(distance <-1,one, distance)
# distance[0][3][2]=31
# distance[0][1][3]=12
# distance[0][0][3]=9
# distance[0][2][3]=-1
# ##处理距离
# protein_pair=distance
# def get_pair_dis_one_hot(pair_dis, bin_size=2, bin_min=-2, bin_max=35):
#     # without compute_mode='donot_use_mm_for_euclid_dist' could lead to wrong result.
#     # pair_dis = torch.cdist(d, d, compute_mode='donot_use_mm_for_euclid_dist')
#     pair_dis[pair_dis>bin_max] = bin_max
#     pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
#     pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=20)
#     print(pair_dis_bin_index)
#     return pair_dis_one_hot
# protein_pair_embedding = Linear(16, c)
# protein_pair = get_pair_dis_one_hot(distance, bin_size=2, bin_min=-2, bin_max=35)
# print(1)
# protein_pair = protein_pair_embedding(protein_pair.float())