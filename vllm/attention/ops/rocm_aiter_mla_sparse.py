# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from functools import lru_cache

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
import vllm.envs as envs

logger = init_logger(__name__)

@triton.jit
def _fp16_mqa_logits_kernel(
    Q_ptr,              # [seq_len, H, D]
    KV_ptr,             # [seq_len_kv, D]
    weights_ptr,        # [seq_len, H]
    cu_start_ptr,       # [seq_len]
    cu_end_ptr,         # [seq_len]
    logits_ptr,         # [seq_len, seq_len_kv]
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    # strides
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.int64,
    stride_logits_k: tl.int64,
    # block sizes
    BLOCK_Q: tl.constexpr,  # Process multiple queries at once
    BLOCK_KV: tl.constexpr,
):
    # 1. Program ID handles a block of Queries now
    pid = tl.program_id(0)
    start_q = pid * BLOCK_Q
    
    # 2. Setup Query Offsets [BLOCK_Q]
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    mask_q = offs_q < seq_len

    # 3. Load Causal Mask boundaries for this block of Queries
    # [BLOCK_Q] vector
    qs_starts = tl.load(cu_start_ptr + offs_q, mask=mask_q, other=0)
    qs_ends = tl.load(cu_end_ptr + offs_q, mask=mask_q, other=0)

    # Optimization: Skip KV blocks that are completely outside the window for this Q block
    # (Optional, but helps performance on diagonal)
    # For simplicity, we process all and mask later, similar to original.

    # 4. Loop over KV Cache in chunks
    # We iterate 0..seq_len_kv
    # For each KV chunk, we compute results for ALL heads and ALL queries in this block
    for kv_start in range(0, seq_len_kv, BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)
        
        # Load KV Transposed: [HEAD_SIZE, BLOCK_KV]
        # We need (D, B) layout for dot product
        d_inds = tl.arange(0, HEAD_SIZE)
        kv_ptrs = (
            KV_ptr 
            + (offs_kv[None, :] * stride_kv_s) # col step
            + (d_inds[:, None] * stride_kv_d)  # row step
        )
        # Mask for tail KV blocks
        mask_kv = offs_kv[None, :] < seq_len_kv
        kv_block = tl.load(kv_ptrs, mask=mask_kv, other=0.0)

        # Accumulator for Logits: [BLOCK_Q, BLOCK_KV]
        acc = tl.zeros([BLOCK_Q, BLOCK_KV], dtype=tl.float32)

        # 5. Inner Loop: Iterate over Heads
        # We process heads sequentially to save memory (Registers/LDS)
        for h in range(NUM_HEADS):
            # Load Q Tile for this Head: [BLOCK_Q, HEAD_SIZE]
            q_ptrs = (
                Q_ptr 
                + (offs_q[:, None] * stride_q_s) 
                + (h * stride_q_h) 
                + (d_inds[None, :] * stride_q_d)
            )
            q_tile = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)

            # Load Weights for this Head: [BLOCK_Q]
            w_ptrs = weights_ptr + (offs_q * stride_w_s) + (h * stride_w_h)
            w_tile = tl.load(w_ptrs, mask=mask_q, other=0.0)

            # Compute Dot: [BLOCK_Q, D] @ [D, BLOCK_KV] -> [BLOCK_Q, BLOCK_KV]
            score = tl.dot(q_tile, kv_block)
            
            # Apply ReLU and Weight
            score = tl.maximum(score, 0.0)
            score = score * w_tile[:, None] # broadcast weight along KV dim

            # Accumulate
            acc += score

        # 6. Apply Causal Masking
        # Check if KV indices are within [start, end) for each Q
        # qs_starts is [BLOCK_Q], offs_kv is [BLOCK_KV]
        # Broadcast comparisons
        in_window = (offs_kv[None, :] >= qs_starts[:, None]) & \
                    (offs_kv[None, :] < qs_ends[:, None])
        
        # 7. Store Logits
        logits_ptrs = (
            logits_ptr 
            + (offs_q[:, None] * stride_logits_s) 
            + (offs_kv[None, :] * stride_logits_k)
        )
        
        # Mask out-of-bounds Q (tail) and out-of-bounds KV (tail) AND causal window
        final_mask = mask_q[:, None] & mask_kv & in_window
        tl.store(logits_ptrs, acc, mask=final_mask)

def fp16_mqa_logits(
    Q,
    KV,
    weights,
    cu_starts,
    cu_ends,
):
    """
    This function computes the logits to be used by a topk function for sparse attention.

    Q:           [seq_len, NUM_HEADS, HEAD_SIZE], dtype float16
    KV:          [seq_len_kv, HEAD_SIZE], dtype float16
    weights:     [seq_len, NUM_HEADS], dtype float32
    cu_starts:   [seq_len], dtype int32, start indices
    cu_ends:     [seq_len], dtype int32, end indices

    Returns:
    logits:      [seq_len, seq_len_kv], dtype float32 (must be initialized to -inf, because of causal masking)
    """
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    # TODO: Currently assuming num_heads and head_size is power of 2.
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."
    # Initialize with -inf because of causal masking
    logits = torch.full(
        (seq_len, seq_len_kv),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=Q.device,
    )

    stride_q_s, stride_q_h, stride_q_d = Q.stride()
    stride_kv_s, stride_kv_d = KV.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    # BLOCK_Q=16: Processes 16 queries at once. Reduces memory loads by 16x.
    # BLOCK_KV=64: Keeps LDS usage safe (~32KB total).
    BLOCK_Q = 16
    BLOCK_KV = 64

    # Grid is now smaller because we handle BLOCK_Q items per program
    grid = (triton.cdiv(seq_len, BLOCK_Q),)

    _fp16_mqa_logits_kernel[grid](
        Q_ptr=Q,
        KV_ptr=KV,
        weights_ptr=weights,
        cu_start_ptr=cu_starts,
        cu_end_ptr=cu_ends,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=stride_q_s,
        stride_q_h=stride_q_h,
        stride_q_d=stride_q_d,
        stride_kv_s=stride_kv_s,
        stride_kv_d=stride_kv_d,
        stride_w_s=stride_w_s,
        stride_w_h=stride_w_h,
        stride_logits_s=stride_logits_s,
        stride_logits_k=stride_logits_k,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        num_warps=4,
        num_stages=1,     # Must be 1. Inner loop handles logic.
        waves_per_eu=1,   # Conservative for safety on MI50 and best value possible for perf
    )

    return logits

# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    kv, scale = kv
    seq_len_kv = kv.shape[0]
    k = kv.to(torch.bfloat16)
    q = q.to(torch.bfloat16)

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits

# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp16_mqa_logits_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. fp16
        kv: `kv` has shape [N, D] with dtype `torch.float16` 
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    k = kv.float()
    q = q.float()
    
    seq_len_kv = kv.shape[0]
    num_q, num_heads, head_dim = q.shape

    # TODO: Make HEAD_CHUNK_SIZE as env variable if this non optimized vibe coded pytorch ops is still used in future build
    # Because HEAD_CHUNK_SIZE is also used in vllm/model_executor/models/deepseek_v2.py - sparse_attn_indexer_fake for the profile run to get correct memory usage
    # TUNE HEAD_CHUNK_SIZE according to AVAILABLE GPU VRAM 
    # 1 heads * 2048 tokens * 2 (peak factor) * 32k context * 4 bytes ~= 0.5 GB memory peak / GPU.
    HEAD_CHUNK_SIZE = 1 

    # 1. Pre-allocate output
    final_logits = torch.full(
        (num_q, seq_len_kv), 
        float("-inf"), 
        device=q.device, 
        dtype=torch.float32
    )

    # 2. Prepare Mask (Broadcasting later)
    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    # Accumulator
    weighted_sum = torch.zeros((num_q, seq_len_kv), device=q.device, dtype=torch.float32)

    # Permute for easier slicing: [H, M, D]
    q_per_head = q.permute(1, 0, 2) 
    weights_per_head = weights.t() # [H, M]
    k_t = k.t() # [D, N]

    # 3. Chunked Loop
    for i in range(0, num_heads, HEAD_CHUNK_SIZE):
        end = min(i + HEAD_CHUNK_SIZE, num_heads)
        
        # Slice the chunk: Shape [Chunk_Size, M, D]
        q_chunk = q_per_head[i:end] 
        w_chunk = weights_per_head[i:end].unsqueeze(-1) # [Chunk, M, 1]

        # Matmul: [Chunk, M, D] @ [D, N] -> [Chunk, M, N]
        # This is the heavy lifting.
        score_chunk = torch.matmul(q_chunk, k_t)
        
        score_chunk = torch.relu(score_chunk)
        
        # Weighted sum: Sum over the chunk dimension (dim 0)
        # [Chunk, M, N] * [Chunk, M, 1] -> [Chunk, M, N] -> Sum -> [M, N]
        chunk_sum = (score_chunk * w_chunk).sum(dim=0)
        
        weighted_sum.add_(chunk_sum)
        
        # Explicit delete to encourage memory freeing before next chunk
        del score_chunk
        del chunk_sum

    # 4. Final Masking
    final_logits = torch.where(mask, weighted_sum, final_logits)

    return final_logits


def rocm_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor]|torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`. Or FP16 ([N, D])
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """

    # TODO(ganyi): Temporarily workaround, will remove the module check and reference
    # path after aiter merge this kernel into main
    @lru_cache
    def has_mqa_logits_module():
        return importlib.util.find_spec("aiter.ops.triton.fp8_mqa_logits") is not None

    if envs.VLLM_ROCM_USE_AITER and has_mqa_logits_module() and not envs.VLLM_ROCM_USE_AITER_MLA_SPARSE_FP16:
        from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
        kv, scale = kv
        return fp8_mqa_logits(q, kv, scale, weights, cu_seqlen_ks, cu_seqlen_ke)
    elif not envs.VLLM_ROCM_USE_AITER and not envs.VLLM_ROCM_USE_AITER_MLA_SPARSE_FP16:
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    elif not envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MLA_SPARSE_FP16:
        return fp16_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    elif envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MLA_SPARSE_FP16:
        return fp16_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)

# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    from vllm.utils.math_utils import cdiv

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device="cuda"
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits

# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp16_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):

    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full([batch_size * next_n, max_model_len], float('-inf'), device=q.device, dtype=torch.float32)
    context_lens = context_lens.tolist()
    
    is_context_lens_2d = False # assumed to never be 2d
   
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.full((next_n, ), context_len, device='cuda', dtype=torch.int32) if is_context_lens_2d \
                    else torch.arange(context_len - next_n, context_len, device='cuda')
        
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        
        num_blocks = (context_len + block_size - 1) // block_size
        block_idxs = block_tables[i][:num_blocks]
        kv_slice = kv_cache[block_idxs]                 # [num_blocks, block_size, kv_heads, dim]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(kv_slice.size(2), dim, -1)    # [kv_heads, dim, total_tokens]
        qx = q[i].transpose(0, 1)                       # q[i]: [next_n, heads, dim] -> [heads, next_n, dim]
        s = torch.matmul(qx, kx).to(logits.dtype)       # [heads, next_n, dim] @ [1, dim, total_tokens] -> [heads, next_n, total_tokens] in fp32 here

        total_len = num_blocks * block_size
        k_offsets = torch.arange(0, total_len, device=q.device)
        mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
        s = torch.where(mask[None, :, :], s, float('-inf'))     # mask shape: [1, next_n, total_tokens]
        s = torch.relu(s) * weight_slice[..., None]             # weight_slice: [heads, next_n] -> [heads, next_n, 1]
        s = s.sum(dim=0)                                        # [next_n, total_tokens]
        logits[i * next_n:(i + 1) * next_n, :total_len] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float('-inf'))

    return logits


def rocm_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller, or native fp16
        kv_cache: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale. Or FP16 : [num_blocks, block_size, 1, D]
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    batch_size, next_n, heads, _ = q.shape
    
    if envs.VLLM_ROCM_USE_AITER and not envs.VLLM_ROCM_USE_AITER_MLA_SPARSE_FP16:
        from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits_stage1

        out_qk = torch.full(
            (heads, batch_size * next_n, max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        deepgemm_fp8_paged_mqa_logits_stage1(
            q,
            kv_cache,
            weights,
            out_qk,
            context_lens,
            block_tables,
            max_model_len,
        )
        return out_qk.sum(dim=0)
    elif not envs.VLLM_ROCM_USE_AITER and not envs.VLLM_ROCM_USE_AITER_MLA_SPARSE_FP16:
        return fp8_paged_mqa_logits_torch(q, kv_cache, weights, context_lens, block_tables, max_model_len)
    elif not envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MLA_SPARSE_FP16:
        return fp16_paged_mqa_logits_torch(q, kv_cache, weights, context_lens, block_tables, max_model_len)
    elif envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MLA_SPARSE_FP16:
        out_qk = torch.full(
            (heads, batch_size * next_n, max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        deepgemm_fp16_paged_mqa_logits_stage1(
            q,
            kv_cache,
            weights,
            out_qk,
            context_lens,
            block_tables,
            max_model_len,
        )
        return out_qk.sum(dim=0)

@triton.jit
def _deepgemm_fp16_paged_mqa_logits_stage1(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,           # FP16
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,          # FP16 [NumBlocks, BlockSize, Heads, Dim]
    stride_k_blk,       # Stride(0): Jump to next physical block (numel per block)
    stride_k_tok,       # Stride(1): Jump to next token inside block (usually Heads*Dim)
    context_len_ptr,
    block_table,        # [Batch, MaxNumBlocks] - Note: This is now a BLOCK table, not token indices
    weights,            # [Batch*NextN, Heads] FP32
    stride_w_batch,
    Out_buffer,
    stride_out_heads,
    stride_out_batch,
    max_model_len,
    max_num_blocks,     # Changed from max_blk_len (tokens) to max_num_blocks
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr, # Must match BlockSize (e.g., 64)
    HiddenDim: tl.constexpr,
    SplitKV: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_length = tl.load(context_len_ptr + pid_batch)

    # Grid logic
    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)
    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
        
    # Cap the length to the assigned split
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    # 1. Load Q (FP16)
    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )

    # 2. Load Weights (FP32)
    # Based on reference: weight_slice = weights[i * next_n:(i + 1) * next_n, ...]
    # These act as gating/scaling factors after ReLU.
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + tl.arange(0, ChunkQ)
    )

    # Loop over KV blocks
    # CRITICAL CHANGE: We iterate aligned to ChunkK (64)
    # We assume ChunkK == BlockSize for this implementation
    for context_idx in range(
        split_context_start, split_context_start + split_context_length, ChunkK
    ):
        # 3. Paged Attention Indexing
        # a. Calculate which logical block we are in
        current_logical_block = context_idx // ChunkK
        
        # b. Load the Physical Block ID from the Block Table
        #    block_table shape: [Batch, MaxNumBlocks]
        physical_block_id = tl.load(
            block_table + pid_batch * max_num_blocks + current_logical_block
        )

        mask_kv = context_idx + tl.arange(0, ChunkK) < context_length

        # 4. Load K (FP16)
        #    Pointer = Base of Physical Block + Offset for Token inside Block
        #    KV_buffer shape is [NumPhysicalBlocks, BlockSize, 1, Dim]
        k = tl.load(
            KV_buffer
            + physical_block_id * stride_k_blk      # Jump to the correct physical block
            + tl.arange(0, ChunkK)[:, None] * stride_k_tok # Iterate tokens 0..63 inside that block
            + tl.arange(0, HiddenDim)[None, :],     # Iterate dim
            mask=mask_kv[:, None],
            other=0.0,
        )

        # 5. Dot Product (FORCE FP32 ACCUMULATION)
        o = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # 6. Activation (ReLU)
        # Matches reference: s = torch.relu(s)
        o = tl.maximum(o, 0.0)

        # 7. Apply Weights (Gating)
        # Matches reference: s = s * weight_slice
        # scale_weight shape is [ChunkQ], o is [ChunkQ, ChunkK]
        # We broadcast scale_weight to apply per-head scaling
        o = o * scale_weight[None, :].T

        # 8. Masking (Causal/Padding)
        mask = (
            context_idx + tl.arange(0, ChunkK) <= context_length - next_n + pid_next_n
        )
        o = tl.where(mask[None, :], o, float("-inf"))

        # Store Output
        tl.store(
            Out_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + (pid_q_head * ChunkQ + tl.arange(0, ChunkQ)[:, None, None])
            * stride_out_heads
            + (context_idx + tl.arange(0, ChunkK)[None, None, :]),
            o[:, None, :],
        )
        

def deepgemm_fp16_paged_mqa_logits_stage1(
    q: torch.Tensor,           # [Batch, NextN, Heads, Dim] (FP16)
    kv_cache: torch.Tensor,    # [NumBlocks, BlockSize, 1, Dim] (FP16)
    weights: torch.Tensor,     # [Batch * NextN, Heads] (FP32)
    out_qk: torch.Tensor,      # Output Logits (FP32)
    context_lens: torch.Tensor,
    block_tables: torch.Tensor, # [Batch, MaxNumBlocks] (Not per-token as blockSize !=1)
    max_model_len: int,
    # MI50 TUNED DEFAULTS:
    ChunkQ: int = 16,           # Smaller tiles = More thread blocks = Higher Occupancy on MI50's 60 CUs.
    ChunkK: int = 64,           # Must match BlockSize, 64 has slightly better performance than 32 (with ChunkQ=32).
    TotalCuCount: int = 60,     # MI50 has 60 CU
    WavePerEU: int = 1,         # Stability
    num_warps: int = 4,   # Sweet spot for register usage on gfx906.
    num_stages: int = 1,  # Essential for stability (avoids LDS crashes)
):
    # Validation
    assert q.dtype == torch.float16
    assert kv_cache.dtype == torch.float16
    
    # Check Block Size matches ChunkK
    block_size = kv_cache.size(1) 
    assert block_size == ChunkK, f"Kernel requires BlockSize ({block_size}) == ChunkK ({ChunkK})"
    
    batch_size, next_n, heads, hidden_dim = q.size()
    _, max_num_blocks = block_tables.size()

    # Calculate strides for the KV Cache
    # KV Cache: [NumBlocks, BlockSize, 1, Dim]
    stride_k_blk = kv_cache.stride(0)  # Steps to jump 1 physical block
    stride_k_tok = kv_cache.stride(1)  # Steps to jump 1 token inside a block

    TileQCount = batch_size * next_n * (heads // ChunkQ)
    SplitKV = (max(1, TotalCuCount // TileQCount) + 4) // 5 * 5 * WavePerEU

    config = {
        "ChunkQ": ChunkQ,
        "ChunkK": ChunkK,
        "HiddenDim": hidden_dim,
        "SplitKV": SplitKV,
    }
    
    grid = (batch_size * next_n * (heads // config["ChunkQ"] * SplitKV),)
    
    _deepgemm_fp16_paged_mqa_logits_stage1[grid](
        batch_size,
        next_n,
        heads,
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache,
        stride_k_blk,  # Stride per block
        stride_k_tok,  # Stride per token
        context_lens,
        block_tables,  # Block Table
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        max_num_blocks, # Max Blocks, not tokens
        num_warps=num_warps,
        num_stages=num_stages,
        **config,
    )
    