# filename: softmax_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, 
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes softmax probabilities for rows in a 2D tensor using Triton.

    Args:
        output_ptr (tl.tensor): Output tensor pointer [batch_size, dim_size] 
        input_ptr (tl.tensor): Input tensor pointer [batch_size, dim_size]
        input_row_stride (int): Stride between rows in input tensor
        output_row_stride (int): Stride between rows in output tensor
        n_cols (int): Number of columns per row
        BLOCK_SIZE (tl.constexpr): Block size (must be power of 2 >= n_cols)

    Returns:
        None: Results are stored in output_ptr

    Explanation:
        1. Get row index from program ID
        2. Calculate input/output row start positions
        3. Load row data with boundary protection mask
        4. Compute max value for numerical stability
        5. Subtract max and compute exponentials
        6. Calculate sum of exponentials
        7. Normalize and store results
    """
    pid = tl.program_id(0)
    row_start_input = pid * input_row_stride
    row_start_output = pid * output_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row data with mask
    x = tl.load(input_ptr + row_start_input + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute numerical stability components
    row_max = tl.max(x, axis=0)
    x_centered = x - row_max
    exp_values = tl.exp(x_centered)
    row_sum = tl.sum(exp_values, axis=0)
    
    # Compute and store softmax
    softmax_output = exp_values / row_sum
    tl.store(output_ptr + row_start_output + col_offsets, softmax_output, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-accelerated softmax implementation for 2D tensors.

    Args:
        x (torch.Tensor): Input tensor [batch_size, dim_size]

    Returns:
        torch.Tensor: Softmax probabilities [batch_size, dim_size]
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    batch_size, n_cols = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (batch_size,)
    
    softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of softmax.

    Args:
        x (torch.Tensor): Input tensor [batch_size, dim_size]

    Returns:
        torch.Tensor: Softmax probabilities [batch_size, dim_size]
    """
    return torch.softmax(x, dim=-1)

if __name__ == "__main__":
    from benchmark import verify_correctness, benchmark_performance

    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        return torch.randn(batch_size, dim_size, device=device, dtype=dtype)

    verify_correctness(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
