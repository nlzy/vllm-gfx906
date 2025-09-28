# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import get_max_shared_memory_bytes

logger = init_logger(__name__)
float8_info = torch.finfo(current_platform.fp8_dtype())


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(query_start_len_ptr, target_idx, num_seqs,
                 BLOCK_Q: tl.constexpr, use_q_block_mode: tl.constexpr):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.autotune(
    configs=[triton.Config({}, num_stages=1, num_warps=2)],
    key=[]
)
@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    BLOCK_FRAGMENT: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    ACTUAL_BLOCK_SIZE: tl.int32,
    USE_FRAGMENT_LOOP: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs,
                           BLOCK_Q, True)

    q_block_start_idx = tl.load(query_start_len_ptr +
                                seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv
    query_offset = (query_offset_0[:, None] * query_stride_0 +
                    query_offset_1[:, None] * query_stride_1 + offs_d[None, :])

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                              mask=query_mask_1,
                              other=0.0)

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
                            )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (
        BLOCK_M - 1) // num_queries_per_kv + 1

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles (blocks) that need to be processed to
    # cover the longest sequence prefix (due to causal masking, blocks beyond
    # this prefix can be skipped)
    num_blocks = cdiv_fn(max_seq_prefix_len, ACTUAL_BLOCK_SIZE)

    # iterate through tiles
    for j in range(0, num_blocks):

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        if not USE_FRAGMENT_LOOP:
            offs_n = tl.arange(0, BLOCK_SIZE)

            v_offset = (physical_block_idx * stride_v_cache_0 +
                        kv_head_idx * stride_v_cache_2 +
                        offs_d[None, :] * stride_v_cache_3 +
                        offs_n[:, None] * stride_v_cache_1)

            k_offset = (physical_block_idx * stride_k_cache_0 +
                        kv_head_idx * stride_k_cache_2 +
                        offs_d[:, None] * stride_k_cache_3 +
                        offs_n[None, :] * stride_k_cache_1)

            # K : (HEAD_SIZE, BLOCK_SIZE)
            K_load = tl.load(key_cache_ptr + k_offset,
                             mask=dim_mask[:, None],
                             other=0.0)

            if K_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    K = K_load
                else:
                    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
            else:
                K = K_load

            # V : (BLOCK_SIZE, HEAD_SIZE)
            V_load = tl.load(value_cache_ptr + v_offset,
                             mask=dim_mask[None, :],
                             other=0.0)

            if V_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    V = V_load
                else:
                    V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
            else:
                V = V_load

            seq_offset = j * ACTUAL_BLOCK_SIZE + offs_n

            seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

            # S : (BLOCK_M, BLOCK_SIZE)
            S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)

            S += scale * tl.dot(Q, K)

            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)

            S = tl.where(query_mask_1[:, None] & query_mask_0[:, None]
                         & seq_mask,
                         S,
                         float("-inf"))

            if SLIDING_WINDOW > 0:
                S = tl.where((context_len + query_pos[:, None] - seq_offset)
                             < SLIDING_WINDOW,
                             S,
                             float("-inf"))

            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset - context_len)

            if USE_QQ_BIAS:
                key_rel_pos = seq_offset - context_len
                is_query_key = (key_rel_pos >= 0) & (key_rel_pos <
                                                     qq_bias_stride_0)
                qq_bias = tl.load(
                    qq_bias_row_ptrs + key_rel_pos[None, :],
                    mask=is_query_key[None, :],
                    other=0.0,
                )
                S += qq_bias

            m_j = tl.maximum(M, tl.max(S, axis=1))
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

            P = tl.exp(S - m_j[:, None])

            l_j = tl.sum(P, axis=1)

            alpha = tl.exp(M - m_j)

            acc = acc * alpha[:, None]

            L = L * alpha + l_j
            M = m_j

            acc += tl.dot(P.to(V.dtype), V)

        else:
            chunk_start = 0
            while chunk_start < ACTUAL_BLOCK_SIZE:
                offs_n = chunk_start + tl.arange(0, BLOCK_FRAGMENT)
                block_mask = offs_n < ACTUAL_BLOCK_SIZE
                mask_has_values = tl.sum(block_mask, axis=0) != 0

                if mask_has_values:
                    offs_n_i64 = offs_n.to(tl.int64)

                    v_offset = (physical_block_idx * stride_v_cache_0 +
                                kv_head_idx * stride_v_cache_2 +
                                offs_d[None, :] * stride_v_cache_3 +
                                offs_n_i64[:, None] * stride_v_cache_1)

                    k_offset = (physical_block_idx * stride_k_cache_0 +
                                kv_head_idx * stride_k_cache_2 +
                                offs_d[:, None] * stride_k_cache_3 +
                                offs_n_i64[None, :] * stride_k_cache_1)

                    # K : (HEAD_SIZE, BLOCK_FRAGMENT)
                    K_load = tl.load(
                        key_cache_ptr + k_offset,
                        mask=dim_mask[:, None] & block_mask[None, :],
                        other=0.0,
                    )

                    if K_load.dtype.is_fp8():
                        if Q.dtype.is_fp8():
                            K = K_load
                        else:
                            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(
                                Q.dtype)
                    else:
                        K = K_load

                    # V : (BLOCK_FRAGMENT, HEAD_SIZE)
                    V_load = tl.load(
                        value_cache_ptr + v_offset,
                        mask=block_mask[:, None] & dim_mask[None, :],
                        other=0.0,
                    )

                    if V_load.dtype.is_fp8():
                        if Q.dtype.is_fp8():
                            V = V_load
                        else:
                            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(
                                Q.dtype)
                    else:
                        V = V_load

                    seq_offset = j * ACTUAL_BLOCK_SIZE + offs_n

                    seq_mask = block_mask[None, :] & (seq_offset[None, :] <
                                                      context_len +
                                                      query_pos[:, None] + 1)

                    # S : (BLOCK_M, BLOCK_FRAGMENT)
                    S = tl.zeros(shape=(BLOCK_M, BLOCK_FRAGMENT),
                                 dtype=tl.float32)

                    S += scale * tl.dot(Q, K)

                    S = tl.where(block_mask[None, :], S, float("-inf"))

                    if USE_SOFTCAP:
                        S = apply_softcap(S, softcap)

                    S = tl.where(query_mask_1[:, None] & query_mask_0[:, None]
                                 & seq_mask,
                                 S,
                                 float("-inf"))

                    if SLIDING_WINDOW > 0:
                        S = tl.where((context_len + query_pos[:, None] -
                                      seq_offset) < SLIDING_WINDOW,
                                     S,
                                     float("-inf"))

                    if USE_ALIBI_SLOPES:
                        S += alibi_slope[:, None] * (seq_offset - context_len)

                    if USE_QQ_BIAS:
                        key_rel_pos = seq_offset - context_len
                        is_query_key = block_mask & (key_rel_pos >= 0) & \
                            (key_rel_pos < qq_bias_stride_0)
                        qq_bias = tl.load(
                            qq_bias_row_ptrs + key_rel_pos[None, :],
                            mask=is_query_key[None, :],
                            other=0.0,
                        )
                        S += qq_bias

                    m_j = tl.maximum(M, tl.max(S, axis=1))
                    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

                    P = tl.exp(S - m_j[:, None])

                    l_j = tl.sum(P, axis=1)

                    alpha = tl.exp(M - m_j)

                    acc = acc * alpha[:, None]

                    L = L * alpha + l_j
                    M = m_j

                    acc += tl.dot(P.to(V.dtype), V)

                chunk_start += BLOCK_FRAGMENT

    # epilogue
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (query_offset_0[:, None] * output_stride_0 +
                     query_offset_1[:, None] * output_stride_1 +
                     offs_d[None, :])

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.autotune(
    configs=[triton.Config({}, num_stages=1, num_warps=2)],
    key=[]
)
@triton.jit
def kernel_unified_attention_3d(
        segm_output_ptr,
        # [num_tokens, num_query_heads, num_segments, head_size]
        segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
        segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
        value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
        sink_ptr,  # [num_query_heads]
        block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
        seq_lens_ptr,  # [num_seqs]
        alibi_slopes_ptr,  # [num_query_heads]
        qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
        scale,  # float32
        k_scale,  # float32
        v_scale,  # float32
        softcap,  # float32
        num_query_heads: tl.constexpr,  # int
        num_queries_per_kv: tl.constexpr,  # int
        block_table_stride: tl.int64,  # int
        query_stride_0: tl.int64,  # int
        query_stride_1: tl.int64,  # int, should be equal to head_size
        qq_bias_stride_0: tl.int64,  # int
        BLOCK_SIZE: tl.constexpr,  # int
        BLOCK_FRAGMENT: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        USE_ALIBI_SLOPES: tl.constexpr,  # bool
        USE_QQ_BIAS: tl.constexpr,  # bool
        USE_SOFTCAP: tl.constexpr,  # bool
        USE_SINKS: tl.constexpr,  # bool
        SLIDING_WINDOW: tl.constexpr,  # int
        stride_k_cache_0: tl.int64,  # int
        stride_k_cache_1: tl.int64,  # int
        stride_k_cache_2: tl.int64,  # int
        stride_k_cache_3: tl.constexpr,  # int
        stride_v_cache_0: tl.int64,  # int
        stride_v_cache_1: tl.int64,  # int
        stride_v_cache_2: tl.int64,  # int
        stride_v_cache_3: tl.constexpr,  # int
        query_start_len_ptr,  # [num_seqs+1]
        BLOCK_Q: tl.constexpr,  # int
        num_seqs: tl.int32,
        BLOCK_M: tl.constexpr,  # int
        NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
        ACTUAL_BLOCK_SIZE: tl.int32,
        USE_FRAGMENT_LOOP: tl.constexpr,  # bool
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs,
                           BLOCK_Q, True)

    q_block_start_idx = tl.load(query_start_len_ptr +
                                seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    blocks_per_segment = cdiv_fn(seq_len, num_segments * ACTUAL_BLOCK_SIZE)

    if segm_idx * blocks_per_segment * ACTUAL_BLOCK_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv

    query_offset = (query_offset_0[:, None] * query_stride_0 +
                    query_offset_1[:, None] * query_stride_1 + offs_d[None, :])

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                              mask=query_mask_1,
                              other=0.0)

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
                            )  # shape: [BLOCK_M]

    num_blocks = cdiv_fn(seq_len, ACTUAL_BLOCK_SIZE)

    # iterate through tiles within current segment
    for j in range(
            segm_idx * blocks_per_segment,
            min((segm_idx + 1) * blocks_per_segment, num_blocks),
    ):
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        if not USE_FRAGMENT_LOOP:
            offs_n = tl.arange(0, BLOCK_SIZE)

            v_offset = (physical_block_idx * stride_v_cache_0 +
                        kv_head_idx * stride_v_cache_2 +
                        offs_d[None, :] * stride_v_cache_3 +
                        offs_n[:, None] * stride_v_cache_1)

            k_offset = (physical_block_idx * stride_k_cache_0 +
                        kv_head_idx * stride_k_cache_2 +
                        offs_d[:, None] * stride_k_cache_3 +
                        offs_n[None, :] * stride_k_cache_1)

            K_load = tl.load(key_cache_ptr + k_offset,
                             mask=dim_mask[:, None],
                             other=0.0)

            if K_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    K = K_load
                else:
                    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
            else:
                K = K_load

            V_load = tl.load(value_cache_ptr + v_offset,
                             mask=dim_mask[None, :],
                             other=0.0)

            if V_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    V = V_load
                else:
                    V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
            else:
                V = V_load

            seq_offset = j * ACTUAL_BLOCK_SIZE + offs_n

            seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

            S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)
            S += scale * tl.dot(Q, K)

            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)

            mask = query_mask_1[:, None] & query_mask_0[:, None] & seq_mask
            S = tl.where(mask, S, float("-inf"))

            if SLIDING_WINDOW > 0:
                S = tl.where((context_len + query_pos[:, None] - seq_offset)
                             < SLIDING_WINDOW,
                             S,
                             float("-inf"))

            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset - context_len)

            if USE_QQ_BIAS:
                key_rel_pos = seq_offset - context_len
                is_query_key = (key_rel_pos >= 0) & (key_rel_pos <
                                                     qq_bias_stride_0)
                qq_bias = tl.load(
                    qq_bias_row_ptrs + key_rel_pos[None, :],
                    mask=is_query_key[None, :],
                    other=0.0,
                )
                S += qq_bias

            m_j = tl.maximum(M, tl.max(S, axis=1))
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

            P = tl.exp(S - m_j[:, None])

            l_j = tl.sum(P, axis=1)

            alpha = tl.exp(M - m_j)

            acc = acc * alpha[:, None]

            L = L * alpha + l_j
            M = m_j

            acc += tl.dot(P.to(V.dtype), V)

        else:
            chunk_start = 0
            while chunk_start < ACTUAL_BLOCK_SIZE:
                offs_n = chunk_start + tl.arange(0, BLOCK_FRAGMENT)
                block_mask = offs_n < ACTUAL_BLOCK_SIZE
                mask_has_values = tl.sum(block_mask, axis=0) != 0

                if mask_has_values:
                    offs_n_i64 = offs_n.to(tl.int64)

                    v_offset = (physical_block_idx * stride_v_cache_0 +
                                kv_head_idx * stride_v_cache_2 +
                                offs_d[None, :] * stride_v_cache_3 +
                                offs_n_i64[:, None] * stride_v_cache_1)

                    k_offset = (physical_block_idx * stride_k_cache_0 +
                                kv_head_idx * stride_k_cache_2 +
                                offs_d[:, None] * stride_k_cache_3 +
                                offs_n_i64[None, :] * stride_k_cache_1)

                    K_load = tl.load(
                        key_cache_ptr + k_offset,
                        mask=dim_mask[:, None] & block_mask[None, :],
                        other=0.0,
                    )

                    if K_load.dtype.is_fp8():
                        if Q.dtype.is_fp8():
                            K = K_load
                        else:
                            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(
                                Q.dtype)
                    else:
                        K = K_load

                    V_load = tl.load(
                        value_cache_ptr + v_offset,
                        mask=block_mask[:, None] & dim_mask[None, :],
                        other=0.0,
                    )

                    if V_load.dtype.is_fp8():
                        if Q.dtype.is_fp8():
                            V = V_load
                        else:
                            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(
                                Q.dtype)
                    else:
                        V = V_load

                    seq_offset = j * ACTUAL_BLOCK_SIZE + offs_n

                    seq_mask = block_mask[None, :] & (seq_offset[None, :] <
                                                      context_len +
                                                      query_pos[:, None] + 1)

                    S = tl.zeros(shape=(BLOCK_M, BLOCK_FRAGMENT),
                                 dtype=tl.float32)

                    S += scale * tl.dot(Q, K)

                    S = tl.where(block_mask[None, :], S, float("-inf"))

                    if USE_SOFTCAP:
                        S = apply_softcap(S, softcap)

                    mask = query_mask_1[:, None] & query_mask_0[:, None] & seq_mask
                    S = tl.where(mask, S, float("-inf"))

                    if SLIDING_WINDOW > 0:
                        S = tl.where((context_len + query_pos[:, None] -
                                      seq_offset) < SLIDING_WINDOW,
                                     S,
                                     float("-inf"))

                    if USE_ALIBI_SLOPES:
                        S += alibi_slope[:, None] * (seq_offset - context_len)

                    if USE_QQ_BIAS:
                        key_rel_pos = seq_offset - context_len
                        is_query_key = block_mask & (key_rel_pos >= 0) & \
                            (key_rel_pos < qq_bias_stride_0)
                        qq_bias = tl.load(
                            qq_bias_row_ptrs + key_rel_pos[None, :],
                            mask=is_query_key[None, :],
                            other=0.0,
                        )
                        S += qq_bias

                    m_j = tl.maximum(M, tl.max(S, axis=1))
                    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

                    P = tl.exp(S - m_j[:, None])

                    l_j = tl.sum(P, axis=1)

                    alpha = tl.exp(M - m_j)

                    acc = acc * alpha[:, None]

                    L = L * alpha + l_j
                    M = m_j

                    acc += tl.dot(P.to(V.dtype), V)

                chunk_start += BLOCK_FRAGMENT

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64) *
        (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        segm_idx * HEAD_SIZE_PADDED + tl.arange(0, HEAD_SIZE_PADDED)[None, :])
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (query_offset_0.to(tl.int64) *
                   (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
                   query_offset_1 * NUM_SEGMENTS_PER_SEQ + segm_idx)
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset,
             L,
             mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    #[num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(query_start_len_ptr, query_token_idx, num_seqs,
                           BLOCK_Q, False)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    blocks_per_segment = cdiv_fn(seq_len, num_segments * BLOCK_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, blocks_per_segment * BLOCK_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32)
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                        0).to(tl.int1)

    # load segment maxima
    segm_offset = (query_token_idx.to(tl.int64) *
                   (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
                   query_head_idx * NUM_SEGMENTS_PER_SEQ +
                   tl.arange(0, NUM_SEGMENTS_PER_SEQ))
    segm_max = tl.load(segm_max_ptr + segm_offset,
                       mask=segm_mask,
                       other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset,
                          mask=segm_mask,
                          other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64) *
        (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED +
        tl.arange(0, HEAD_SIZE_PADDED)[None, :])
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (query_token_idx * output_stride_0 +
                     query_head_idx * output_stride_1 +
                     tl.arange(0, HEAD_SIZE_PADDED))
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    block_size = v.shape[1]
    assert q.element_size() >= 2 or block_size >= 32, \
        "Block size must be at least 32 for fp8"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], \
        "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    head_size_padded = triton.next_power_of_2(head_size)

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(
        num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    block_fragment = min(triton.next_power_of_2(block_size), 256)

    if current_platform.is_rocm():
        max_shared = 0
        try:
            max_shared = get_max_shared_memory_bytes()
        except Exception:
            max_shared = 64 * 1024

        if max_shared > 0:
            kv_bytes = max(k.element_size(), v.element_size())
            kv_bytes = max(kv_bytes, 1)

            # conservative shared memory estimate per iteration
            def estimate_shared(fragment: int) -> int:
                shared = 0
                shared += 2 * head_size_padded * fragment * kv_bytes
                shared += 2 * BLOCK_M * fragment * 4
                return shared

            # keep block_fragment as power of two while respecting limit
            while (block_fragment > 32 and
                   estimate_shared(block_fragment) > max_shared):
                block_fragment //= 2

    use_fragment_loop = block_fragment < block_size

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    if current_platform.is_rocm():
        # Keep the faster 2D kernel for very small decode batches to avoid the
        # expensive segmented 3D compilation, but fall back to 3D once there is
        # enough parallelism or context length to benefit from it.
        ROCM_MIN_DECODE_TOKENS_FOR_3D = 4
        ROCM_MIN_CONTEXT_FOR_3D = 256
        long_context = max_seqlen_k >= ROCM_MIN_CONTEXT_FOR_3D
        decode_ready = (max_seqlen_q <= 1
                        and total_num_q_blocks * num_kv_heads <= 128
                        and (q.shape[0] >= ROCM_MIN_DECODE_TOKENS_FOR_3D
                             or long_context))
        use_3d_kernel = decode_ready
    else:
        use_3d_kernel = (max_seqlen_q <= 1 and
                         total_num_q_blocks * num_kv_heads <= 128)

    # if batch contains a prefill
    if not use_3d_kernel:
        kernel_unified_attention_2d[(
            total_num_q_blocks,
            num_kv_heads,
        )](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            out_scale=1 / output_scale if output_scale is not None else 1.0,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            BLOCK_FRAGMENT=block_fragment,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
            ACTUAL_BLOCK_SIZE=block_size,
            USE_FRAGMENT_LOOP=use_fragment_loop,
        )
    else:
        # for initial version, NUM_SEGMENTS = 16 is chosen as a default
        # value that showed good performance in tests
        NUM_SEGMENTS = 16

        segm_output = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            head_size_padded,
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )

        kernel_unified_attention_3d[(
            total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
                query_ptr=q,
                key_cache_ptr=k,
                value_cache_ptr=v,
                sink_ptr=sinks,
                block_tables_ptr=block_table,
                seq_lens_ptr=seqused_k,
                alibi_slopes_ptr=alibi_slopes,
                qq_bias_ptr=qq_bias,
                scale=softmax_scale,
                k_scale=k_descale,
                v_scale=v_descale,
                softcap=softcap,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                block_table_stride=block_table.stride(0),
                query_stride_0=q.stride(0),
                query_stride_1=q.stride(1),
                qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
                BLOCK_SIZE=block_size,
                BLOCK_FRAGMENT=block_fragment,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=head_size_padded,
                USE_ALIBI_SLOPES=use_alibi_slopes,
                USE_QQ_BIAS=use_qq_bias,
                USE_SOFTCAP=(softcap > 0),
                USE_SINKS=(sinks is not None),
                SLIDING_WINDOW=(1 + window_size[0]),
                stride_k_cache_0=k.stride(0),
                stride_k_cache_1=k.stride(1),
                stride_k_cache_2=k.stride(2),
                stride_k_cache_3=k.stride(3),
                stride_v_cache_0=v.stride(0),
                stride_v_cache_1=v.stride(1),
                stride_v_cache_2=v.stride(2),
                stride_v_cache_3=v.stride(3),
                query_start_len_ptr=cu_seqlens_q,
                BLOCK_Q=BLOCK_Q,
                num_seqs=num_seqs,
                BLOCK_M=BLOCK_M,
                NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
                ACTUAL_BLOCK_SIZE=block_size,
                USE_FRAGMENT_LOOP=use_fragment_loop,
            )

        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 /
            output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
            USE_FP8=output_scale is not None,
        )
