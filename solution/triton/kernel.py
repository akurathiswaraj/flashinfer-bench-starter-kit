import math
import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_prefill_fullseq(
    Q, K, V, S, A_log, a, dt_bias, b, CU_SEQLENS,
    Out, NS,
    sq0, sq1, sq2,
    sk0, sk1, sk2,
    sv0, sv1, sv2,
    ss0, ss1, ss2, ss3,
    so0, so1, so2,
    sn0, sn1, sn2, sn3,
    sa0, sa1,
    sb0, sb1,
    scale,
    GQA_RATIO: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    hid     = tl.program_id(1)
    vid     = tl.program_id(2)
    qhid    = hid // GQA_RATIO

    d     = tl.arange(0, D)
    vr    = vid * BV + tl.arange(0, BV)
    vmask = vr < D

    start = tl.load(CU_SEQLENS + seq_idx).to(tl.int64)
    end   = tl.load(CU_SEQLENS + seq_idx + 1).to(tl.int64)
    seq_len = end - start

    st_ptrs = S + seq_idx * ss0 + hid * ss1 + vr[:, None] * ss2 + d[None, :] * ss3
    st = tl.load(st_ptrs, mask=vmask[:, None], other=0.0).to(tl.float32)

    dt_v = tl.load(dt_bias + hid).to(tl.float32)
    Al_v = tl.load(A_log + hid).to(tl.float32)
    exp_Al = tl.exp(Al_v)

    for t_rel in range(0, seq_len):
        t_abs = start + t_rel

        a_v = tl.load(a + t_abs * sa0 + hid * sa1).to(tl.float32)
        b_v = tl.load(b + t_abs * sb0 + hid * sb1).to(tl.float32)
        x   = a_v + dt_v
        sp  = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
        g   = tl.exp(-exp_Al * sp)
        beta = 1.0 / (1.0 + tl.exp(-b_v))

        k_vec = tl.load(K + t_abs * sk0 + qhid * sk1 + d * sk2).to(tl.float32)
        q_vec = tl.load(Q + t_abs * sq0 + qhid * sq1 + d * sq2).to(tl.float32)
        qk    = tl.sum(q_vec * k_vec, axis=0)
        v_in  = tl.load(V + t_abs * sv0 + hid * sv1 + vr * sv2, mask=vmask, other=0.0).to(tl.float32)

        state_k = tl.sum(st * k_vec[None, :], axis=1)
        state_q = tl.sum(st * q_vec[None, :], axis=1)
        dv    = beta * (v_in - g * state_k)
        out_v = scale * (g * state_q + qk * dv)
        st = g * st + dv[:, None] * k_vec[None, :]

        tl.store(Out + t_abs * so0 + hid * so1 + vr * so2, out_v.to(tl.bfloat16), mask=vmask)

    tl.store(st_ptrs, st, mask=vmask[:, None])
    ns_ptrs = NS + seq_idx * sn0 + hid * sn1 + vr[:, None] * sn2 + d[None, :] * sn3
    tl.store(ns_ptrs, st, mask=vmask[:, None])


@torch.no_grad()
def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    total_len, Hq, D = q.shape
    Hv  = v.shape[1]
    GQA = Hv // Hq
    num_seqs = cu_seqlens.shape[0] - 1

    assert D == 128

    if scale is None or float(scale) == 0.0:
        scale = 1.0 / math.sqrt(D)
    scale = float(scale)

    dev = q.device

    A_c  = A_log.reshape(-1)[-Hv:].contiguous().float()
    dt_c = dt_bias.reshape(-1)[-Hv:].contiguous().float()
    q_c  = q.contiguous()
    k_c  = k.contiguous()
    v_c  = v.contiguous()
    a_c  = a.contiguous()
    b_c  = b.contiguous()
    cu_c = cu_seqlens.contiguous()

    if state is None:
        state_c = torch.zeros(num_seqs, Hv, D, D, dtype=torch.float32, device=dev)
    else:
        state_c = state.clone().contiguous().float()

    output = torch.empty(total_len, Hv, D, dtype=torch.bfloat16, device=dev)
    ns_t   = torch.empty(num_seqs, Hv, D, D, dtype=torch.float32, device=dev)

    BV = 64
    grid = (num_seqs, Hv, triton.cdiv(D, BV))

    _gdn_prefill_fullseq[grid](
        q_c, k_c, v_c, state_c, A_c, a_c, dt_c, b_c, cu_c,
        output, ns_t,
        *q_c.stride(), *k_c.stride(), *v_c.stride(),
        *state_c.stride(),
        *output.stride(), *ns_t.stride(),
        *a_c.stride(), *b_c.stride(),
        scale,
        GQA_RATIO=GQA, D=D, BV=BV,
        num_warps=4, num_stages=2,
    )

    return output, ns_t

