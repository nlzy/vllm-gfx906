# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm import envs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    # Shuffle weight along the last dimension so that
    # we folded the weights to adjance location
    # Example:
    # input:
    #       [[1, 2, 3, 4, 5, 6],
    #        [7, 8, 9, 10, 11, 12]]
    # output:
    #       [[1, 4, 2, 5, 3, 6],
    #        [7, 10, 8, 11, 9, 12]]
    # This will be used together with triton swiglu kernel
    shape = w.shape
    N = shape[-1]
    first = w[..., :N // 2]
    second = w[..., N // 2:]

    stacked = torch.stack((first, second), dim=-1)
    w_shuffled = stacked.reshape(shape)
    return w_shuffled


def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=3, num_warps=2),
    ]

def get_heuristics():
    return {
        'BLOCK_SIZE_M': lambda args: min(16, triton.next_power_of_2(args['M']))
    }

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K']
)
@triton.heuristics(values=get_heuristics())
@triton.jit
def triton_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def triton_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions" # NOTE(gfx906): b.shape inv
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N, K = b.shape # NOTE(gfx906): b.shape inv
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    triton_matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(1), b.stride(0),  # NOTE(gfx906): b.stride inv
        c.stride(0), c.stride(1),  #
    )
    return c

def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                    output_tokens_tensor: torch.Tensor,
                    presence_penalties: torch.Tensor,
                    frequency_penalties: torch.Tensor,
                    repetition_penalties: torch.Tensor) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts 
        are padded to the maximum prompt length within the batch using 
        `vocab_size` as the padding value. The value `vocab_size` is used 
        for padding because it does not correspond to any valid token ID 
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(prompt_tokens_tensor,
                                                   vocab_size, num_seqs)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties
    apply_repetition_penalties(logits, prompt_mask, output_mask,
                               repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def default_unquantized_gemm(layer: torch.nn.Module,
                             x: torch.Tensor,
                             weight: torch.Tensor,
                             bias: Optional[torch.Tensor] = None):
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    use_skinny = (x.dtype in [torch.float16, torch.bfloat16] \
                  and bias is None)
    if use_skinny is not True:
        return torch.nn.functional.linear(x, weight, bias)

    x_view = x.view(-1, x.size(-1))
    n = x_view.shape[0]
    m = weight.shape[0]
    k = weight.shape[1]

    # prefer skinny GEMV kernel
    if m % 4 == 0 and n == 1 and k <= 8192 and k % 8 == 0:
        out = ops.LLMM1(weight, x_view, 4)
        return out.view(*x.shape[:-1], weight.shape[0])

    # low batch size, use triton matmul
    if n <= 16:
        return triton_matmul(x, weight)

    # otherwise, use native torch
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_impl_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(layer: torch.nn.Module,
                          x: torch.Tensor,
                          weight: torch.Tensor,
                          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.vllm.rocm_unquantized_gemm_impl(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm_impl",
    op_func=rocm_unquantized_gemm_impl,
    mutates_args=[],
    fake_impl=rocm_unquantized_gemm_impl_fake,
    dispatch_key=current_platform.dispatch_key,
)


def cpu_unquantized_gemm(layer: torch.nn.Module,
                         x: torch.Tensor,
                         weight: torch.Tensor,
                         bias: Optional[torch.Tensor] = None):
    if getattr(layer, "use_cpu_sgl", False):
        return torch.ops._C.weight_packed_linear(x, weight, bias, True)
    else:
        return torch.nn.functional.linear(x, weight, bias)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    else:
        return default_unquantized_gemm
