import torch

@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """将世界系三维向量 v 投影至四元数 q 所代表的机身局部坐标系 (Body Frame)"""
    q_w = q[:, 0]
    q_vec = q[:, 1:4]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(-1, 1, 3), v.view(-1, 3, 1)).squeeze(-1) * 2.0
    return a - b + c

@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """将机身局部坐标系向量 v 转换到世界坐标系"""
    q_w = q[:, 0]
    q_vec = q[:, 1:4]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(-1, 1, 3), v.view(-1, 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """将角度映射至 [-pi, pi] 区间"""
    angles %= 2.0 * torch.pi
    angles -= 2.0 * torch.pi * (angles > torch.pi)
    return angles
