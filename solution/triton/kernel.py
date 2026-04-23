import math
import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_fused(
    # inputs
    Q, K, V, S, A_log, a, dt_bias, b,
    # outputs
    Out, NS,

    # Q strides: [B, Hq, D]
    sq0, sq1, sq2,
    # K strides: [B, Hk, D]
    sk0, sk1, sk2,
    # V strides: [B, Hv, D]
    sv0, sv1, sv2,
    # S strides: [B, Hv, D, D]
    ss0, ss1, ss2, ss3,
    # Out strides: [B, Hv, D]
    so0, so1, so2,
    # NS strides: [B, Hv, D, D]
    sn0, sn1, sn2, sn3,
    # a strides: [B, Hv]
    sa0, sa1,
    # b strides: [B, Hv]
    sb0, sb1,
    scale,
    GQA: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
):
    bid = tl.program_id(0)   # batch
    hid = tl.program_id(1)   # v-head
    vid = tl.program_id(2)   # tile index along V/D dimension

    qh = hid // GQA
    kh = hid // GQA

    d = tl.arange(0, D)
    vr = vid * BV + tl.arange(0, BV)
    vmask = vr < D

    # scalar gates
    a_v  = tl.load(a + bid * sa0 + hid * sa1).to(tl.float32)
    b_v  = tl.load(b + bid * sb0 + hid * sb1).to(tl.float32)
    dt_v = tl.load(dt_bias + hid).to(tl.float32)   # flattened [Hv]
    Al_v = tl.load(A_log + hid).to(tl.float32)     # flattened [Hv]

    # stable softplus
    x = a_v + dt_v
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    g = tl.exp(-tl.exp(Al_v) * sp)
    beta = 1.0 / (1.0 + tl.exp(-b_v))

    # q and k vectors [D]
    k_vec = tl.load(K + bid * sk0 + kh * sk1 + d * sk2).to(tl.float32)
    q_vec = tl.load(Q + bid * sq0 + qh * sq1 + d * sq2).to(tl.float32)
    qk = tl.sum(q_vec * k_vec, axis=0)

    # v tile [BV]
    v_in = tl.load(
        V + bid * sv0 + hid * sv1 + vr * sv2,
        mask=vmask,
        other=0.0
    ).to(tl.float32)

    # state tile [BV, D], treating state as [B, Hv, D, D]
    st_ptrs = (
        S
        + bid * ss0
        + hid * ss1
        + vr[:, None] * ss2
        + d[None, :] * ss3
    )
    st = tl.load(st_ptrs, mask=vmask[:, None], other=0.0).to(tl.float32)

    # recurrence
    old_v = g * tl.sum(st * k_vec[None, :], axis=1)                       # [BV]
    dv = beta * (v_in - old_v)                                            # [BV]
    out = scale * (g * tl.sum(st * q_vec[None, :], axis=1) + qk * dv)    # [BV]
    nst = g * st + dv[:, None] * k_vec[None, :]                           # [BV, D]

    # store output
    tl.store(
        Out + bid * so0 + hid * so1 + vr * so2,
        out.to(tl.bfloat16),
        mask=vmask
    )

    # store new state
    ns_ptrs = (
        NS
        + bid * sn0
        + hid * sn1
        + vr[:, None] * sn2
        + d[None, :] * sn3
    )
    tl.store(ns_ptrs, nst, mask=vmask[:, None])


@torch.no_grad()
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale):
    B, T, Hq, D = q.shape
    _, _, Hk, Dk = k.shape
    _, _, Hv, Dv = v.shape

    assert T == 1
    assert D == 128 and Dk == 128 and Dv == 128
    assert Hq == 4 and Hk == 4 and Hv == 8

    if scale is None or float(scale) == 0.0:
        scale = 1.0 / math.sqrt(D)
    scale = float(scale)

    dev = q.device

    # [B, H, D]
    q_c = q.squeeze(1).contiguous()
    k_c = k.squeeze(1).contiguous()
    v_c = v.squeeze(1).contiguous()

    # [B, Hv]
    a_c = a.squeeze(1).contiguous().float()
    b_c = b.squeeze(1).contiguous().float()

    # Flatten to [Hv]
    A_c = A_log.reshape(-1)[-Hv:].contiguous().float()
    dt_c = dt_bias.reshape(-1)[-Hv:].contiguous().float()

    # Force float32 state
    if state is None:
        state_c = torch.zeros((B, Hv, D, D), dtype=torch.float32, device=dev)
    else:
        state_c = state.contiguous().float()

    out_t = torch.empty((B, Hv, D), dtype=torch.bfloat16, device=dev)
    ns_t = torch.empty((B, Hv, D, D), dtype=torch.float32, device=dev)

    BV = 32
    grid = (B, Hv, triton.cdiv(D, BV))

    _gdn_decode_fused[grid](
        q_c, k_c, v_c, state_c, A_c, a_c, dt_c, b_c,
        out_t, ns_t,

        *q_c.stride(),
        *k_c.stride(),
        *v_c.stride(),
        *state_c.stride(),
        *out_t.stride(),
        *ns_t.stride(),
        *a_c.stride(),
        *b_c.stride(),

        scale,
        GQA=Hv // Hq,
        D=D,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )

    return out_t.unsqueeze(1), ns_t
