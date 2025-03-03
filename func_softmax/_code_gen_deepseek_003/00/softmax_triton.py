# filename: softmax_triton.py
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes softmax for a row of elements using Triton.

    Args:
        output_ptr (tl.tensor): Pointer to output tensor [batch_size, n_cols]
        input_ptr (tl.tensor): Pointer to input tensor [batch_size, n_cols]
        n_cols (int): Number of columns in each row
        BLOCK_SIZE (tl.constexpr): Block size for parallel processing

    Returns:
        None: Modifies output_ptr in-place

    Explanation:
        1. Compute row index for current program
        2. Initialize pointers to input/output rows
        3. Find maximum value in row for numerical stability
        4. Compute exponential values adjusted by max
        5. Sum exponential values for normalization
        6. Normalize and store results in output
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_row_ptr = input_ptr + row_idx * n_cols
    row = tl.load(input_row_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Step 1: Find row max
    row_max = tl.max(row, axis=0)
    
    # Step 2: Subtract max and compute exp
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)
    
    # Step 3: Compute sum of exponents
    denominator = tl.sum(numerator, axis=0)
    
    # Step 4: Normalize and store
    softmax_output = numerator / denominator
    output_row_ptr = output_ptr + row_idx * n_cols
    tl.store(output_row_ptr + col_offsets, softmax_output, mask=mask)

def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-accelerated Softmax function.

    Args:
        x (torch.Tensor): Input tensor [batch_size, dim_size]

    Returns:
        torch.Tensor: Softmax activated output [batch_size, dim_size]

    Explanation:
        1. Validate input dimensions
        2. Allocate output tensor
        3. Configure kernel launch parameters
        4. Launch Triton kernel with optimized blocks
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Limit block size for register pressure
    
    y = torch.empty_like(x)
    
    grid = (n_rows,)
    softmax_kernel[grid](
        y, x, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8
    )
    return y

def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of softmax.

    Args:
        x (torch.Tensor): Input tensor [batch_size, dim_size]

    Returns:
        torch.Tensor: Softmax activated output [batch_size, dim_size]

    Explanation:
        1. Apply native PyTorch softmax along last dimension
    """
    return F.softmax(x, dim=-1)

if __name__ == "__main__":
    from benchmark import verify_correctness, benchmark_performance

    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x
    
    # Test correctness
    verify_correctness(softmax_triton, softmax_torch, input_generator, [batch_sizes, dim_sizes])
    
    # Benchmark performance
    benchmark_performance(softmax_triton, softmax_torch, input_generator, [batch_sizes, dim_sizes])
