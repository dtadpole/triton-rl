# filename: softmax_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['dim_size']
)
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    dim_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute softmax over the last dimension of a 2D tensor using Triton.

    Args:
        input_ptr (tl.tensor): Pointer to input tensor [n_rows, dim_size]
        output_ptr (tl.tensor): Pointer to output tensor [n_rows, dim_size]
        dim_size (tl.constexpr): Size of the last dimension
        BLOCK_SIZE (tl.constexpr): Block size for computation

    Returns:
        None: Writes results in-place to output_ptr

    Explanation:
        1. Calculate row index for parallel processing
        2. Initialize max and sum variables
        3. Iterate through elements in blocks to compute row max
        4. Iterate again to compute exponential values and their sum
        5. Final iteration to write normalized softmax values
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Step 1: Find row max for numerical stability
    row_max = tl.full((1,), float('-inf'), tl.float32)
    for offset in range(0, dim_size, BLOCK_SIZE):
        cols = offset + col_offsets
        mask = cols < dim_size
        val = tl.load(input_ptr + row_idx * dim_size + cols, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(val, 0))

    # Step 2: Compute exp(x - max) and sum
    sum_exp = tl.zeros((1,), tl.float32)
    for offset in range(0, dim_size, BLOCK_SIZE):
        cols = offset + col_offsets
        mask = cols < dim_size
        val = tl.load(input_ptr + row_idx * dim_size + cols, mask=mask, other=float('-inf'))
        exp_val = tl.exp(val - row_max)
        tl.store(output_ptr + row_idx * dim_size + cols, exp_val, mask=mask)
        sum_exp += tl.sum(exp_val, 0)

    # Step 3: Normalize by sum_exp
    for offset in range(0, dim_size, BLOCK_SIZE):
        cols = offset + col_offsets
        mask = cols < dim_size
        val = tl.load(output_ptr + row_idx * dim_size + cols, mask=mask)
        normalized = val / sum_exp
        tl.store(output_ptr + row_idx * dim_size + cols, normalized, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Triton softmax implementation wrapper"""
    y = torch.empty_like(x)
    n_rows, dim_size = x.shape
    softmax_kernel[(n_rows,)](x, y, dim_size, BLOCK_SIZE=triton.next_power_of_two(dim_size))
    return y

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference softmax implementation"""
    return torch.nn.functional.softmax(x, dim=1)

if __name__ == "__main__":
    from benchmark import verify_correctness, benchmark_performance

    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x

    verify_correctness(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
