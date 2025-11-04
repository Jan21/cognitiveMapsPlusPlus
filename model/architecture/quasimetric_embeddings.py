from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchqmet

def softplus_inv_float(y: float) -> float:
    threshold: float = 20.  # https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html#torch-nn-functional-softplus
    if y > threshold:
        return y
    else:
        return np.log(np.expm1(y))

class GradMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, mult: Union[float, torch.Tensor]) -> torch.Tensor:
        ctx.mult_is_tensor = isinstance(mult, torch.Tensor)
        if ctx.mult_is_tensor:
            assert not mult.requires_grad
            ctx.save_for_backward(mult)
        else:
            ctx.mult = mult
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.mult_is_tensor:
            mult, = ctx.saved_tensors
        else:
            mult = ctx.mult
        return grad_output * mult, None


def grad_mul(x: torch.Tensor, mult: Union[float, torch.Tensor]) -> torch.Tensor:
    if not isinstance(mult, torch.Tensor) and mult == 0:
        return x.detach()
    return GradMul.apply(x, mult)


class LatentDynamicsLoss(nn.Module):

    def __init__(self, *, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, zx, zy, quasimetric_model, latent_dynamics, actions):
        pred_zy = latent_dynamics(zx, actions)
        distances = quasimetric_model(pred_zy, zy, bidirectional=True)
        sq_dists = distances.square().mean()
        dist_p2n, dist_n2p = distances.unbind(-1)
        return sq_dists * self.weight, sq_dists, dist_p2n.mean(), dist_n2p.mean()


class GlobalPushLoss(nn.Module):

    def __init__(self, *, softplus_beta: float, softplus_offset: float):
        super().__init__()
        self.softplus_beta = softplus_beta
        self.softplus_offset = softplus_offset

    def forward(self, zx, zy, quasimetric_model):
        distances = quasimetric_model(zx, torch.roll(zy, 1, dims=0)) # create random pairs of zx and zy
        normalized_distances = F.softplus(self.softplus_offset - distances, beta=self.softplus_beta) # penalize large distances less.
        return normalized_distances.mean(), distances.mean()


class LocalConstraintLoss(nn.Module):

    def __init__(self, *, epsilon: float, step_cost: float, init_lagrange_multiplier: float):
        super().__init__()
        self.epsilon = epsilon
        self.step_cost = step_cost
        self.init_lagrange_multiplier = init_lagrange_multiplier
        self.raw_lagrange_multiplier = nn.Parameter(
            torch.tensor(softplus_inv_float(init_lagrange_multiplier), dtype=torch.float32))

    def forward(self, zx, zy, quasimetric_model):
        distances = quasimetric_model(zx, zy)
        lagrange_mult = F.softplus(self.raw_lagrange_multiplier)  # make positive
        # lagrange multiplier is minimax training, so grad_mul -1
        lagrange_mult = grad_mul(lagrange_mult, -1)
        sq_deviation = (distances - self.step_cost).relu().square().mean()
        violation = (sq_deviation - self.epsilon ** 2)
        loss = violation * lagrange_mult
        return loss, distances.mean(), sq_deviation, violation, lagrange_mult

##### MODEL ######

class QuasimetricModel(nn.Module):

    def __init__(self, *, input_size: int, hidden_size: int, components: int): 
        super().__init__()
        self.input_size = input_size
        self.quasimetric_head = torchqmet.IQE(hidden_size, hidden_size // components)
        self.projector = MLP(input_size, hidden_size)

    def forward(self, zx, zy):
        #px = self.projector(zx)  # [B x D]
        #py = self.projector(zy)  # [B x D]
        #return self.quasimetric_head(px, py)
        return self.quasimetric_head(zx, zy)



class LatentDynamics(nn.Module):

    def __init__(self, latent_size: int, action_input_size: int, action_latent_size: int):
        super().__init__()
        self.action_embedding = MLP(action_input_size, action_latent_size)
        self.dynamics = MLP(latent_size + action_latent_size, latent_size)

    def forward(self, zx, action):
        action = self.action_embedding(action)
        return self.dynamics(torch.cat([zx, action], dim=-1)) + zx



class QuasimetricEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, latent_size: int, action_input_size: int, action_latent_size: int, hidden_size: int, **kwargs):
        super().__init__()
        self.input_size = vocab_size
        self.latent_size = latent_size
        self.action_input_size = action_input_size
        self.action_latent_size = action_latent_size
        self.hidden_size = hidden_size
        self.quasimetric_model = QuasimetricModel(input_size=latent_size, hidden_size=hidden_size, components=8)
        self.latent_dynamics = LatentDynamics(latent_size, action_input_size, action_latent_size)
        self.global_push_loss = GlobalPushLoss(softplus_beta=1, softplus_offset=30.0)
        self.local_constraint_loss = LocalConstraintLoss(epsilon=0.25, step_cost=1, init_lagrange_multiplier=0.01)
        self.node_embedding = nn.Embedding(vocab_size, latent_size)

    def forward(self, batch):
        x = batch['x']
        y = batch['y']
        action = batch['action']
        zx = self.node_embedding(x)
        zy = self.node_embedding(y)
        result = {"zx": zx, "zy": zy, "action": action}
        return result

    def get_loss(self, result, batch):
        zx, zy, action = result['zx'], result['zy'], result['action']
        global_push_loss, global_push_distance = self.global_push_loss(zx, zy, self.quasimetric_model)
        local_constraint_loss, _, sq_dev, _, _ = self.local_constraint_loss(zx, zy, self.quasimetric_model)
        total_loss = global_push_loss + local_constraint_loss
        result["loss"] = total_loss #+ mse_sum_norm_diffs
        result["mse_sum_norm_diffs"] = sq_dev
        result["logits"] = None
        return result


    def get_param_groups(self):
        param_groups = []
        quasimetric_params = set(self.quasimetric_model.parameters())
        print(f"Quasimetric parameters: {len(quasimetric_params)}")
        param_groups.append({
            'params': list(quasimetric_params),
            'lr': 0.0001,
            #'weight_decay': 0.0,
            'betas': (0.9, 0.999),
        })
        local_constrains_params = set(self.local_constraint_loss.parameters())
        print(f"Local constraints parameters: {len(local_constrains_params)}")
        param_groups.append({
                'params': list(local_constrains_params),
                'lr': 0.01,
                #'weight_decay': 0.0,
                'betas': (0.9, 0.999),
            })
        return param_groups


class MLP(nn.Module):
    def __init__(self, input_dim: int, d_latent: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_latent)
        self.linear = nn.Linear(d_latent, d_latent)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        # for i in range(5):
        #     residual = x
        #     x = self.activation(self.linear(x))
        #     x = x + residual
        return x