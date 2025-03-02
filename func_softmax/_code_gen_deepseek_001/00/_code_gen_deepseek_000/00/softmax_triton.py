# filename: softmax_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr,  # Pointer to the input tensor
    y_ptr,  # Pointer to the output tensor
    dim_size: int,  # Size of the softmax dimension
    BLOCK_SIZE: tl.constexpr,  # Block size (must be power of 2)
):
    """
    Computes softmax for a row using Triton.

    Args:
        x_ptr (tl.tensor): Input tensor pointer [dim_size]
        y_ptr (tl.tensor): Output tensor pointer [dim_size]
        dim_size (int): Size of the softmax dimension
        BLOCK_SIZE (tl.constexpr): Block size for computation

    Returns:
        None: Output is stored through y_ptr

    Explanation:
        1. Program ID identifies the row index
        2. Load row data from global memory
        3. Compute numerical stable maximum
        4. Normalize and exponentiate values
        5. Write results back to global memory
    """
    row_idx = tl.program_id(0)
    offsets = row_idx * dim_size + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < dim_size

    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    x_minus_max = x - tl.max(x, axis=0)
    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    tl.store(y_ptr + offsets, softmax_output, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-accelerated softmax implementation for 2D tensors.

    Args:
        x (torch.Tensor): Input tensor [batch_size, dim_size]

    Returns:
        torch.Tensor: Softmax output [batch_size, dim_size]

    Explanation:
        1. Validate input dimensions
        2. Allocate output tensor
        3. Determine optimal block size
        4. Launch kernel with appropriate grid size
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    n_rows, dim_size = x.shape
    y = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(dim_size)
    grid = (n_rows,)
    
    softmax_kernel[grid](x, y, dim_size, BLOCK_SIZE=BLOCK_SIZE)
    return y

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of softmax.

    Args:
        x (torch.Tensor): Input tensor [batch_size, dim_size]

    Returns:
        torch.Tensor: Softmax output [batch_size, dim_size]
    """
    return torch.nn.functional.softmax(x, dim=-1)

batch_sizes = [256, 512, 1024, 2048]
dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
    x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
    return x

if __name__ == "__main__":
    from benchmark import verify_correctness, benchmark_performance
    
    verify_correctness(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
