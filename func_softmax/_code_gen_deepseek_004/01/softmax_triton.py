# filename: softmax_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes softmax for each row of a 2D matrix using Triton.

    Args:
        output_ptr (tensor): Pointer to output tensor [batch_size, dim_size]
        input_ptr (tensor): Pointer to input tensor [batch_size, dim_size]
        n_cols (int): Number of columns (dim_size) in each row
        BLOCK_SIZE (tl.constexpr): Tile size for kernel execution

    Returns:
        None: Output is stored in output_ptr

    Explanation:
        1. Find maximum value in each row for numerical stability
        2. Compute exponential values adjusted by row maximum
        3. Calculate sum of exponentials for each row
        4. Normalize exponential values by sum to get probabilities
    """
    pid = tl.program_id(0)
    row_start = pid * n_cols
    
    # Phase 1: Compute row maximum
    row_max = -float('inf')
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        input_vals = tl.load(input_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(input_vals, axis=0))

    # Phase 2: Compute exponential sum
    sum_exp = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        input_vals = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)
        exp_vals = tl.exp(input_vals - row_max)
        sum_exp += tl.sum(exp_vals, axis=0)

    # Phase 3: Compute final normalized values
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        input_vals = tl.load(input_ptr + row_start + offsets, mask=mask)
        output_vals = tl.exp(input_vals - row_max) / sum_exp
        tl.store(output_ptr + row_start + offsets, output_vals, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies softmax using Triton kernel to 2D input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size]

    Returns:
        torch.Tensor: Softmax output tensor of shape [batch_size, dim_size]

    Explanation:
        1. Validates input tensor dimensions
        2. Determines optimal block size for kernel execution
        3. Launches Triton kernel with configured grid and block parameters
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    
    output = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](output, x, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    return output

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies standard PyTorch softmax to input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size]

    Returns:
        torch.Tensor: Softmax output tensor of shape [batch_size, dim_size]

    Explanation:
        1. Uses PyTorch's built-in softmax implementation
        2. Applies along last dimension (-1)
    """
    return torch.softmax(x, dim=-1)

# Testing and benchmarking setup
batch_sizes = [256, 512, 1024, 2048]
dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
    return torch.randn(batch_size, dim_size, device=device, dtype=dtype)

from benchmark import verify_correctness_func, benchmark_performance_func

# Perform verification and benchmarking
if __name__ == "__main__":
    verify_correctness_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
