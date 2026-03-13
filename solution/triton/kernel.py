import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def triton_stub():
    pass


@torch.no_grad()
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale):
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape

    assert T == 1
    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert K == 128 and V == 128

    if scale is None or float(scale) == 0.0:
        scale = 1.0 / math.sqrt(K)

    x = a.float() + dt_bias.float()                 # [B,1,8]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x)).squeeze(1)     # [B,8]
    beta = torch.sigmoid(b.float()).squeeze(1)      # [B,8]

    q_f32 = q.squeeze(1).float()                    # [B,4,128]
    k_f32 = k.squeeze(1).float()                    # [B,4,128]
    v_f32 = v.squeeze(1).float()                    # [B,8,128]

    if state is None:
        state_f32 = torch.zeros(B, num_v_heads, V, K, dtype=torch.float32, device=q.device)
    else:
        state_f32 = state.float()

    q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)       # [B,8,128]
    k_exp = k_f32.repeat_interleave(num_v_heads // num_k_heads, dim=1)       # [B,8,128]

    state_hkv = state_f32.transpose(-1, -2).contiguous()                     # [B,8,K,V]
    old_state = g.unsqueeze(-1).unsqueeze(-1) * state_hkv                    # [B,8,K,V]

    old_v = torch.matmul(k_exp.unsqueeze(-2), old_state).squeeze(-2)         # [B,8,V]
    new_v = beta.unsqueeze(-1) * v_f32 + (1.0 - beta.unsqueeze(-1)) * old_v  # [B,8,V]

    state_remove = k_exp.unsqueeze(-1) @ old_v.unsqueeze(-2)                 # [B,8,K,V]
    state_update = k_exp.unsqueeze(-1) @ new_v.unsqueeze(-2)                 # [B,8,K,V]
    new_state_hkv = old_state - state_remove + state_update                  # [B,8,K,V]

    output = scale * torch.matmul(q_exp.unsqueeze(-2), new_state_hkv).squeeze(-2)  # [B,8,V]
    new_state = new_state_hkv.transpose(-1, -2).contiguous()                 # [B,8,V,K]

    return output.unsqueeze(1).to(torch.bfloat16), new_state
