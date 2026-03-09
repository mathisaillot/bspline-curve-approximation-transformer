from bspline_curve_approximation_transformer.utils import TRITON_AVAILABLE

import torch

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    BLOCK_M_fwd = 64
    BLOCK_N_fwd = 128
    num_stages_fwd = 2
    num_warps_fwd = 4

    BLOCK_M_bwd = 64
    BLOCK_N_bwd = 64
    num_stages_bwd = 1
    num_warps_bwd = 4

    BLOCK_M_fwd_padding = 64
    BLOCK_N_fwd_padding = 128
    num_stages_fwd_padding = 1
    num_warps_fwd_padding = 8

    BLOCK_M_bwd_padding = 64
    BLOCK_N_bwd_padding = 64
    num_stages_bwd_padding = 2
    num_warps_bwd_padding = 4

    BLOCK_M_bwd_padding_32 = 64
    BLOCK_N_bwd_padding_32 = 32
    num_stages_bwd_padding_32 = 2
    num_warps_bwd_padding_32 = 8


    @triton.jit
    def _attn_fwd_inner(acc, l_i, m_i, q,
                        K_ptr, V_ptr, stride_v,  #
                        qk_scale,
                        BLOCK_M: tl.constexpr,
                        BLOCK_N: tl.constexpr,
                        N_CTX: tl.constexpr, ):
        # loop over k, v and update accumulator
        for start_m in range(0, N_CTX, BLOCK_M):
            # -- compute qk ----
            k = tl.load(K_ptr)
            qk = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
            qk += tl.dot(q, k)

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            # -- update output accumulator --
            acc = acc * alpha[:, None]
            v = tl.load(V_ptr)
            acc = tl.dot(p, v, acc)
            # update m_i and l_i
            m_i = m_ij
            V_ptr += BLOCK_M * stride_v
            K_ptr = tl.advance(K_ptr, (0, BLOCK_M))
        return acc, l_i, m_i


    @triton.jit
    def _attn_fwd_inner_padding(acc, l_i, m_i, q,  #
                                K, V, K_offset, V_offset, stride,  #
                                qk_scale, end_m, mask_m,  #
                                BLOCK_N: tl.constexpr,
                                BLOCK_M: tl.constexpr,  #
                                N_CTX: tl.constexpr):
        # loop over k, v and update accumulator
        it = end_m // BLOCK_M
        for m in range(0, it):
            # -- compute qk ----
            k = tl.load(K + K_offset)
            qk = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
            qk += tl.dot(q, k)

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            # -- update output accumulator --
            acc = acc * alpha[:, None]
            # update acc
            v = tl.load(V + V_offset)
            acc = tl.dot(p, v, acc)
            # update m_i and l_i
            m_i = m_ij
            V_offset += BLOCK_M * stride
            K_offset += BLOCK_M * stride

        m_lefts = end_m - it * BLOCK_M
        if m_lefts > 0:
            # -- compute qk ----
            # mask = tl.arange(0, BLOCK_M) > m_lefts
            # mask = mask[None, :]
            k = tl.load(K + K_offset, mask=mask_m[None, :], other=0.0)
            # qk = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
            qk = tl.dot(q, k) + tl.where(mask_m[None, :], 0.0, float('-inf'))

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

            p = tl.math.exp2(qk)
            # p = tl.where(mask, 0.0, p)
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            # -- update output accumulator --
            acc = acc * alpha[:, None]
            # update acc
            v = tl.load(V + V_offset, mask=mask_m[:, None], other=0.0)
            acc = tl.dot(p, v, acc)
            # update m_i and l_i
            m_i = m_ij

        return acc, l_i, m_i


    @triton.jit
    def _attn_fwd(Q, K, V, qk_scale, M, OUT,  #
                  stride_qz, stride_qh, stride_qm, stride_qk,  #
                  stride_kz, stride_kh, stride_kn, stride_kk,  #
                  stride_vz, stride_vh, stride_vk, stride_vn,  #
                  stride_outz, stride_outh, stride_outm, stride_outn,  #
                  Z, H,  #
                  N_CTX: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  BLOCK_DMODEL: tl.constexpr,  #
                  BLOCK_N: tl.constexpr,  #
                  ):
        start_n = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        out_offset = off_z.to(tl.int64) * stride_outz + off_h.to(tl.int64) * stride_outh

        index_DMODEL = tl.arange(0, BLOCK_DMODEL)
        off_m = tl.arange(0, BLOCK_M)

        V_ptr = V + v_offset + (off_m[:, None] * stride_vk + index_DMODEL[None, :] * stride_vn)

        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(start_n * BLOCK_N, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_M),
            order=(0, 1),
        )
        out_block_ptr = tl.make_block_ptr(
            base=OUT + out_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_outm, stride_outn),
            offsets=(start_n * BLOCK_N, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )
        # initialize offsets
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_N], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)

        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                        K_block_ptr, V_ptr, stride_vk,  #
                                        qk_scale,  #
                                        BLOCK_M, BLOCK_N,  #
                                        N_CTX  #
                                        )
        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_n
        tl.store(m_ptrs, m_i)
        tl.store(out_block_ptr, acc.to(OUT.type.element_ty))


    @triton.jit
    def _attn_fwd_padding(Q, K, V, qk_scale, M, Out,  #
                          padding_lengths,
                          stride_qkvz, stride_qkvh, stride_qkvm, stride_qkvk,  #
                          stride_outz, stride_outh, stride_outm, stride_outk,  #
                          Z, H,  #
                          N_CTX: tl.constexpr,  #
                          BLOCK_M: tl.constexpr,  #
                          BLOCK_DMODEL: tl.constexpr,  #
                          BLOCK_N: tl.constexpr,  #
                          ):
        off_hz = tl.program_id(0)
        off_z = off_hz // H
        off_h = off_hz % H
        qkv_offset = off_z.to(tl.int64) * stride_qkvz + off_h.to(tl.int64) * stride_qkvh
        out_offset = off_z.to(tl.int64) * stride_outz + off_h.to(tl.int64) * stride_outh

        index_DMODEL = tl.arange(0, BLOCK_DMODEL)
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)

        padding = tl.load(padding_lengths + off_z)

        it = padding // BLOCK_N
        it_m = padding // BLOCK_M

        V_offset = (off_m[:, None] * stride_qkvm + index_DMODEL[None, :] * stride_qkvk) + qkv_offset
        K_offset = (off_m[None, :] * stride_qkvm + index_DMODEL[:, None] * stride_qkvk) + qkv_offset
        Q_offset = (off_n[:, None] * stride_qkvm + index_DMODEL[None, :] * stride_qkvk) + qkv_offset
        OUT_offset = (off_n[:, None] * stride_outm + index_DMODEL[None, :] * stride_outk) + out_offset

        n_offset = off_hz * N_CTX + off_n

        n_lefts = padding - it * BLOCK_N
        m_lefts = padding - it_m * BLOCK_M

        mask_n = off_n < n_lefts
        mask_m = off_m < m_lefts

        for m in range(0, it):
            # initialize pointer to m and l
            m_i = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_N], dtype=tl.float32) + 1.0
            acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

            # load q: it will stay in SRAM throughout
            q = tl.load(Q_offset + Q)

            acc, l_i, m_i = _attn_fwd_inner_padding(acc, l_i, m_i, q,
                                                    K, V, K_offset, V_offset, stride_qkvm,  #
                                                    qk_scale, padding, mask_m,
                                                    BLOCK_N, BLOCK_M, N_CTX  #
                                                    )
            # epilogue
            m_i += tl.math.log2(l_i)
            acc = acc / l_i[:, None]
            tl.store(M + n_offset, m_i)
            tl.store(Out + OUT_offset, acc)

            Q_offset += BLOCK_N * stride_qkvm
            OUT_offset += BLOCK_N * stride_outm
            n_offset += BLOCK_N

        if n_lefts > 0:
            # tl.device_print('pad', n_lefts)
            m_i = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_N], dtype=tl.float32) + 1.0
            acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

            # load q: it will stay in SRAM throughout
            # tl.device_print("offset ", tl.max(OUT_offset))
            # tl.device_print("mask ", offset_mask_out)
            q = tl.load(Q + Q_offset, mask=mask_n[:, None], other=float("-inf"))

            acc, l_i, m_i = _attn_fwd_inner_padding(acc, l_i, m_i, q,
                                                    K, V, K_offset, V_offset, stride_qkvm,  #
                                                    qk_scale, padding, mask_m,
                                                    BLOCK_N, BLOCK_M, N_CTX  #
                                                    )
            # epilogue
            m_i += tl.math.log2(l_i)
            acc = acc / l_i[:, None]
            tl.store(M + n_offset, m_i, mask=mask_n)
            tl.store(Out + OUT_offset, acc, mask=mask_n[:, None])


    @triton.jit
    def _attn_bwd_preprocess(O, DO,  #
                             Delta,  #
                             stride_outz, stride_outh, stride_outm, stride_outn,
                             Z, H, N_CTX,  #
                             BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                             ):
        off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        off_n = tl.arange(0, HEAD_DIM)
        out_offset = (off_z.to(tl.int64) * stride_outz + off_h.to(tl.int64) * stride_outh +
                      (off_m[:, None] * stride_outm + off_n[None, :] * stride_outn))
        # load
        o = tl.load(O + out_offset)
        do = tl.load(DO + out_offset).to(tl.float32)
        delta = tl.sum(o * do, axis=1)
        # write-back
        tl.store(Delta + off_hz * N_CTX + off_m, delta)


    @triton.jit
    def _attn_bwd_preprocess_padding(O, DO,  #
                                     Delta,  #
                                     stride_outz, stride_outh, stride_outm, stride_outn,
                                     Z, H, N_CTX, padding_lengths,  #
                                     BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                                     ):
        id = tl.program_id(0) * BLOCK_M
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        padding = tl.load(padding_lengths + off_z)
        m_lefts = padding - id
        if m_lefts > 0:
            off_m = tl.arange(0, BLOCK_M)
            off_n = tl.arange(0, HEAD_DIM)
            off_h = off_hz % H
            off_D = off_hz * N_CTX + off_m + id

            offset = (off_z.to(tl.int64) * stride_outz + off_h.to(tl.int64) * stride_outh +
                      (off_m[:, None] + id) * stride_outm + off_n[None, :] * stride_outn)
            if m_lefts >= BLOCK_M:
                # load
                o = tl.load(O + offset)
                do = tl.load(DO + offset).to(tl.float32)
                delta = tl.sum(o * do, axis=1)
                # write-back
                tl.store(Delta + off_D, delta)
            else:
                mask = off_m < m_lefts
                o = tl.load(O + offset, mask=mask[:, None])
                do = tl.load(DO + offset, mask=mask[:, None]).to(tl.float32)
                delta = tl.sum(o * do, axis=1)
                # write-back
                tl.store(Delta + off_D, delta, mask=mask)


    @triton.jit
    def _bwd_kernel(
            Q, K, V, sm_scale, qk_scale,
            DO,
            DQ, DK, DV,
            L, D,
            stride_qkvz, stride_qkvh, stride_qkvm, stride_qkvk,
            stride_dqkvz, stride_dqkvh, stride_dqkvn, stride_dqkvk,
            stride_outz, stride_outh, stride_outm, stride_outk,
            Z, H, N_CTX,
            BLOCK_M: tl.constexpr,
            BLOCK_DMODEL: tl.constexpr,
            BLOCK_N: tl.constexpr,
    ):
        off_hz = tl.program_id(0)
        off_z = (off_hz // H).to(tl.int64)
        off_h = (off_hz % H).to(tl.int64)
        # offset pointers for batch/head
        qkv_offset = (off_z * stride_qkvz + off_h * stride_qkvh).to(tl.int64)
        dqkv_offset = (off_z * stride_dqkvz + off_h * stride_dqkvh).to(tl.int64)
        out_offset = (off_z * stride_outz + off_h * stride_outh).to(tl.int64)
        Q += qkv_offset
        K += qkv_offset
        V += qkv_offset
        DO += out_offset
        DQ += dqkv_offset
        DK += dqkv_offset
        DV += dqkv_offset

        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX

        offs_m = tl.arange(0, BLOCK_M) - BLOCK_M
        offs_dmodel = tl.arange(0, BLOCK_DMODEL)
        offs_n = tl.arange(0, BLOCK_N)
        offs_q = (offs_n[:, None] * stride_qkvm + offs_dmodel[None, :] * stride_qkvk)
        offs_dq = (offs_n[:, None] * stride_dqkvn + offs_dmodel[None, :] * stride_dqkvk)
        offs_out = (offs_n[:, None] * stride_outm + offs_dmodel[None, :] * stride_outk)

        q_incr_ptr = BLOCK_N * stride_qkvm
        dq_incr_ptr = BLOCK_N * stride_dqkvn
        out_incr_ptr = BLOCK_N * stride_outm

        for start_m in range(0, N_CTX // BLOCK_M):
            # initialize row/col offsets
            offs_m += BLOCK_M
            offs_kv = (offs_m[:, None] * stride_qkvm + offs_dmodel[None, :] * stride_qkvk)
            offs_dkv = (offs_m[:, None] * stride_dqkvn + offs_dmodel[None, :] * stride_dqkvk)
            # initialize pointers to value-like data
            q_ptrs = Q + offs_q
            do_ptrs = DO + offs_out
            dq_ptrs = DQ + offs_dq

            # k and v stay in SRAM throughout
            v_ptrs = V + offs_kv
            k_ptrs = K + offs_kv
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            # initialize dv amd dk
            dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
            dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
            # loop over rows
            for start_n in range(0, N_CTX // BLOCK_N):

                offs_n_curr = start_n * BLOCK_N + offs_n
                # load q, k, v, do on-chip
                q = tl.load(q_ptrs)
                l_i = tl.load(l_ptrs + offs_n_curr)
                do = tl.load(do_ptrs)
                Di = tl.load(D_ptrs + offs_n_curr)
                # recompute p = softmax(qk, dim=-1).T
                #    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                qk = tl.dot(q, tl.trans(k))
                qk *= qk_scale

                p = tl.math.exp2(qk - l_i[:, None])
                # compute dv

                dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
                # compute dp = dot(v, do)
                # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
                # dp += tl.dot(do, tl.trans(urv))
                dp = tl.dot(do, tl.trans(v)) - Di[:, None]
                # compute ds = p * (dp - delta[:, None])
                ds = p * dp * sm_scale
                # compute dk = dot(ds.T, q)
                dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
                # compute dq
                if start_m == 0:
                    tl.store(dq_ptrs, tl.dot(ds.to(Q.dtype.element_ty), k))
                else:
                    dq = tl.load(dq_ptrs)
                    dq += tl.dot(ds.to(Q.dtype.element_ty), k)
                    tl.store(dq_ptrs, dq)
                # increment pointers
                dq_ptrs += dq_incr_ptr
                do_ptrs += out_incr_ptr
                q_ptrs += q_incr_ptr

            # write-back
            dv_ptrs = DV + offs_dkv
            dk_ptrs = DK + offs_dkv
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)


    @triton.jit
    def _bwd_inner_padding(
            Q, k, v, sm_scale, qk_scale,
            it_n, n_lefts,
            DO, DQ, dk, dv,
            L, D,
            q_offset, dq_offset, dout_offset,
            n_offset, offs_n, mask_n,
            start_m, stride_qn, stride_dqn, stride_doutn,
            BLOCK_N: tl.constexpr,
    ):
        # loop over rows
        for start_n in range(0, it_n):
            # load q, k, v, do on-chip
            q = tl.load(Q + q_offset)
            do = tl.load(DO + dout_offset)
            l_i = tl.load(L + n_offset)
            Di = tl.load(D + n_offset)
            # recompute p = softmax(qk, dim=-1).T
            #    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, tl.trans(k))
            qk *= qk_scale

            p = tl.math.exp2(qk - l_i[:, None])
            # compute dv
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            # compute dp = dot(v, do)
            # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            # dp += tl.dot(do, tl.trans(urv))
            dp = tl.dot(do, tl.trans(v)) - Di[:, None]
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            # compute dq
            if start_m == 0:
                tl.store(DQ + dq_offset, tl.dot(ds.to(Q.dtype.element_ty), k))
            else:
                dq = tl.load(DQ + dq_offset)
                dq += tl.dot(ds.to(Q.dtype.element_ty), k)
                tl.store(DQ + dq_offset, dq)
            # increment pointers
            q_offset += BLOCK_N * stride_qn
            dq_offset += BLOCK_N * stride_dqn
            dout_offset += BLOCK_N * stride_doutn
            n_offset += BLOCK_N

        if n_lefts > 0:
            # load q, k, v, do on-chip
            q = tl.load(Q + q_offset, mask=mask_n[:, None], other=0.0)
            do = tl.load(DO + dout_offset, mask=mask_n[:, None], other=0.0)
            l_i = tl.load(L + n_offset, mask=mask_n, other=float('inf'))
            Di = tl.load(D + n_offset, mask=mask_n, other=0.0)
            # recompute p = softmax(qk, dim=-1).T
            #    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, tl.trans(k))
            qk *= qk_scale

            p = tl.math.exp2(qk - l_i[:, None])
            # compute dv

            dv += tl.dot(tl.trans(p), do)
            # compute dp = dot(v, do)
            # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            # dp += tl.dot(do, tl.trans(urv))
            dp = tl.dot(do, tl.trans(v)) - Di[:, None]
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds), q)
            # compute dq
            if start_m == 0:
                tl.store(DQ + dq_offset, tl.dot(ds, k), mask=mask_n[:, None])
            else:
                dq = tl.load(DQ + dq_offset, mask=mask_n[:, None])
                dq += tl.dot(ds, k)
                tl.store(DQ + dq_offset, dq, mask=mask_n[:, None])
        return dk, dv


    @triton.jit
    def _bwd_kernel_padding(
            Q, K, V, sm_scale, qk_scale, padding_lengths,
            DO,
            DQ, DK, DV,
            L,
            D,
            stride_qkvz, stride_qkvh, stride_qkvn, stride_qkvk,
            stride_dqkvz, stride_dqkvh, stride_dqkvn, stride_dqkvk,
            stride_outz, stride_outh, stride_outn, stride_outk,
            Z, H, N_CTX,
            BLOCK_M: tl.constexpr,
            BLOCK_DMODEL: tl.constexpr,
            BLOCK_N: tl.constexpr,
    ):
        off_hz = tl.program_id(0)
        off_z = (off_hz // H).to(tl.int64)
        off_h = (off_hz % H).to(tl.int64)

        # offset pointers for batch/head
        qkv_offset = (off_z * stride_qkvz + off_h * stride_qkvh).to(tl.int64)
        dqkv_offset = (off_z * stride_dqkvz + off_h * stride_dqkvh).to(tl.int64)
        out_offset = (off_z * stride_outz + off_h * stride_outh).to(tl.int64)

        padding = tl.load(padding_lengths + off_z)

        offs_m = tl.arange(0, BLOCK_M)
        offs_dmodel = tl.arange(0, BLOCK_DMODEL)
        offs_n = tl.arange(0, BLOCK_N)

        it_m = padding // BLOCK_M
        it_n = padding // BLOCK_N

        n_lefts = padding - it_n * BLOCK_N
        m_lefts = padding - it_m * BLOCK_M

        mask_n = offs_n < n_lefts
        mask_m = offs_m < m_lefts

        q_offset = qkv_offset + (offs_n[:, None] * stride_qkvn + offs_dmodel[None, :] * stride_qkvk)
        dq_offset = dqkv_offset + (offs_n[:, None] * stride_dqkvn + offs_dmodel[None, :] * stride_dqkvk)
        dout_offset = out_offset + (offs_n[:, None] * stride_outn + offs_dmodel[None, :] * stride_outk)
        offs_kv = qkv_offset + (offs_m[:, None] * stride_qkvn + offs_dmodel[None, :] * stride_qkvk)
        offs_dkv = dqkv_offset + (offs_m[:, None] * stride_dqkvn + offs_dmodel[None, :] * stride_dqkvk)
        n_offset = off_hz * N_CTX + offs_n

        for start_m in range(0, it_m):
            # k and v stay in SRAM throughout
            v_ptrs = V + offs_kv
            k_ptrs = K + offs_kv

            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)

            # initialize dv amd dk
            dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
            dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

            # loop over rows
            dk, dv = _bwd_inner_padding(Q, k, v, sm_scale, qk_scale,
                                        it_n, n_lefts,
                                        DO, DQ, dk, dv, L, D,
                                        q_offset, dq_offset, dout_offset,
                                        n_offset, offs_n, mask_n,
                                        start_m, stride_qkvn, stride_dqkvn, stride_outn,
                                        BLOCK_N)

            # write-back
            dv_ptrs = DV + offs_dkv
            dk_ptrs = DK + offs_dkv
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
            offs_kv += BLOCK_M * stride_qkvn
            offs_dkv += BLOCK_M * stride_dqkvn

        if m_lefts > 0:
            # k and v stay in SRAM throughout
            v_ptrs = V + offs_kv
            k_ptrs = K + offs_kv

            k = tl.load(k_ptrs, mask=mask_m[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_m[:, None], other=0.0)

            # initialize dv amd dk
            dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
            dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

            # loop over rows
            dk, dv = _bwd_inner_padding(Q, k, v, sm_scale, qk_scale,
                                        it_n, n_lefts,
                                        DO, DQ, dk, dv, L, D,
                                        q_offset, dq_offset, dout_offset,
                                        n_offset, offs_n, mask_n,
                                        it_m, stride_qkvn, stride_dqkvn, stride_outn,
                                        BLOCK_N)

            # write-back
            dv_ptrs = DV + offs_dkv
            dk_ptrs = DK + offs_dkv
            tl.store(dv_ptrs, dv, mask=mask_m[:, None])
            tl.store(dk_ptrs, dk, mask=mask_m[:, None])


    class flash_attention(torch.autograd.Function):

        @staticmethod
        def forward(ctx, q, k, v, sm_scale, padding_lengths=None):
            assert q.stride() == k.stride() == v.stride()
            # shape constraints
            Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
            BATCH, N_HEAD, N_CTX = q.shape[:3]

            out = torch.zeros((BATCH, N_HEAD, N_CTX, Lq), device=q.device, dtype=torch.float32)

            M_out = torch.zeros((BATCH, N_HEAD, N_CTX), device=q.device, dtype=torch.float32)

            qk_scale = sm_scale * 1.4426950408889634

            # forward
            if padding_lengths is None:
                _attn_fwd[(triton.cdiv(N_CTX, BLOCK_N_fwd), BATCH * N_HEAD, 1)](
                    q, k, v, qk_scale, M_out, out,  #
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),  #
                    BATCH, N_HEAD,  #
                    N_CTX=N_CTX,  #
                    BLOCK_M=BLOCK_M_fwd,  #
                    BLOCK_N=BLOCK_N_fwd,  #
                    BLOCK_DMODEL=Lk,  #
                    num_warps=num_warps_fwd,  #
                    num_stages=num_stages_fwd  #
                )
            else:
                _attn_fwd_padding[(BATCH * N_HEAD,)](
                    q, k, v, qk_scale, M_out, out, padding_lengths,  #
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),  #
                    BATCH, N_HEAD,  #
                    N_CTX=N_CTX,  #
                    BLOCK_M=BLOCK_M_fwd_padding,  #
                    BLOCK_N=BLOCK_N_fwd_padding,  #
                    BLOCK_DMODEL=Lk,  #
                    num_warps=num_warps_fwd_padding,  #
                    num_stages=num_stages_fwd_padding  #
                )

            ctx.save_for_backward(q, k, v, out, M_out)
            ctx.sm_scale = sm_scale
            ctx.qk_scale = qk_scale
            ctx.BLOCK_DMODEL = Lk
            ctx.padding_lengths = padding_lengths
            return out

        def backward(ctx, dout):
            q, k, v, out, M_out = ctx.saved_tensors
            dout = dout.contiguous()
            assert out.stride() == dout.stride()

            dq = torch.zeros_like(q)
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            assert q.stride() == k.stride() == v.stride()
            assert dq.stride() == dk.stride() == dv.stride()

            BATCH, N_HEAD, N_CTX = q.shape[:3]
            HEAD_DIM = ctx.BLOCK_DMODEL
            sm_scale = ctx.sm_scale
            qk_scale = ctx.qk_scale
            padding_lengths = ctx.padding_lengths

            if padding_lengths is None:
                PRE_BLOCK_out = min(128, N_CTX)
                if N_CTX % PRE_BLOCK_out != 0:
                    PRE_BLOCK_out = 32
                delta_out = torch.empty_like(M_out)
                _attn_bwd_preprocess[(N_CTX // PRE_BLOCK_out, BATCH * N_HEAD)](
                    out, dout,  #
                    delta_out,  #
                    dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                    BATCH, N_HEAD, N_CTX,  #
                    BLOCK_M=PRE_BLOCK_out, HEAD_DIM=HEAD_DIM  #
                )

                block_n_out = BLOCK_N_bwd
                block_m_out = 32 if HEAD_DIM > 32 else BLOCK_M_bwd

                _bwd_kernel[(N_HEAD * BATCH,)](
                    q, k, v, sm_scale, qk_scale,
                    dout,
                    dq, dk, dv,
                    M_out, delta_out,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                    q.shape[0], q.shape[1], q.shape[2],
                    BLOCK_M=block_m_out,
                    BLOCK_N=block_n_out,
                    BLOCK_DMODEL=HEAD_DIM,
                    num_warps=num_warps_bwd,
                    num_stages=num_stages_bwd,
                )

            else:
                PRE_BLOCK_out = 64
                # assert N_CTX % PRE_BLOCK_out == 0
                delta_out = torch.empty_like(M_out)
                _attn_bwd_preprocess_padding[(triton.cdiv(N_CTX, PRE_BLOCK_out), BATCH * N_HEAD)](
                    out, dout,  #
                    delta_out,  #
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    BATCH, N_HEAD, N_CTX, padding_lengths,  #
                    BLOCK_M=PRE_BLOCK_out, HEAD_DIM=HEAD_DIM  #
                )
                block_m_out = 32 if HEAD_DIM > 32 else BLOCK_M_bwd

                _bwd_kernel_padding[(N_HEAD * BATCH,)](
                    q, k, v, sm_scale, qk_scale, padding_lengths,
                    dout,
                    dq, dk, dv,
                    M_out, delta_out,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                    q.shape[0], q.shape[1], q.shape[2],
                    BLOCK_M=block_m_out,
                    BLOCK_N=BLOCK_N_bwd_padding_32 if HEAD_DIM > 32 else BLOCK_N_bwd_padding,
                    BLOCK_DMODEL=HEAD_DIM,
                    num_warps=num_warps_bwd_padding_32 if HEAD_DIM > 32 else num_warps_bwd_padding,
                    num_stages=num_stages_bwd_padding_32 if HEAD_DIM > 32 else num_stages_bwd_padding,
                )
            return dq, dk, dv, None, None


    FlashAttention = flash_attention.apply
else:
    FlashAttention = None
