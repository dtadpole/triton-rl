# filename: softmax_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes softmax for a row of the input tensor.

    Args:
        output_ptr (tl.tensor): Pointer to output tensor [batch_size, n_cols]
        input_ptr (tl.tensor): Pointer to input tensor [batch_size, n_cols]
        n_cols (int): Number of columns in each row
        BLOCK_SIZE (tl.constexpr): Block size for kernel execution

    Returns:
        None: Modifies output_ptr in-place

    Explanation:
        1. Determine row index for current block
        2. Load row data from input tensor
        3. Compute row maximum for numerical stability
        4. Subtract max and compute exponentials
        5. Compute sum of exponentials
        6. Normalize and store results
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_ptrs = input_ptr + row_idx * n_cols + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    output_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-accelerated softmax function.

    Args:
        x (torch.Tensor): Input tensor [batch_size, dim_size]

    Returns:
        torch.Tensor: Softmax output [batch_size, dim_size]

    Explanation:
        1. Validate input dimensions
        2. Allocate output tensor
        3. Configure kernel launch parameters
        4. Launch optimized softmax kernel
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](y, x, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference softmax implementation.

    Args:
        x (torch.Tensor): Input tensor [batch_size, dim_size]

    Returns:
        torch.Tensor: Softmax output [batch_size, dim_size]

    Explanation:
        1. Apply native PyTorch softmax along last dimension
    """
    return torch.softmax(x, dim=1)

def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
    """Generate random input tensor with specified dimensions"""
    return torch.randn(batch_size, dim_size, device=device, dtype=dtype)

if __name__ == "__main__":
    from benchmark import verify_correctness, benchmark_performance
    
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    verify_correctness(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
