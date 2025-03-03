# filename: softmax_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    Computes softmax probabilities along the last dimension of a 2D tensor.

    Args:
        output_ptr (triton.Pointer): Output tensor pointer. Shape (batch, dim)
        input_ptr (triton.Pointer): Input tensor pointer. Shape (batch, dim)
        input_row_stride (int): Stride between input tensor rows
        output_row_stride (int): Stride between output tensor rows
        n_cols (int): Number of columns in input tensor
        BLOCK_SIZE (tl.constexpr): Processing block size for kernel optimization

    Returns:
        None: Modifies output tensor in-place

    Explanation:
        1. Each program processes one row using program_id(0) for row index
        2. Iterate through columns in BLOCK_SIZE chunks to find row maximum
        3. Compute exponentials normalized by row maximum
        4. Calculate sum of exponentials across entire row
        5. Write normalized softmax values (exponential / sum) to output
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Find row max
    max_val = -float('inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + col_offsets
        mask = cols < n_cols
        ptr = input_ptr + row_start + cols
        curr = tl.load(ptr, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(curr, axis=0))

    # Compute exp(x - max)
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + col_offsets
        mask = cols < n_cols
        ptr = input_ptr + row_start + cols
        curr = tl.load(ptr, mask=mask, other=0)
        exp_val = tl.exp(curr - max_val)
        sum_exp += tl.sum(exp_val, axis=0)

    # Compute and store softmax
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + col_offsets
        mask = cols < n_cols
        ptr = input_ptr + row_start + cols
        curr = tl.load(ptr, mask=mask, other=0)
        softmax_val = tl.exp(curr - max_val) / sum_exp
        out_ptr = output_ptr + row_start + cols
        tl.store(out_ptr, softmax_val, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies softmax using Triton optimized kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, dim)

    Returns:
        torch.Tensor: Output tensor with softmax applied along last dimension

    Explanation:
        1. Determines optimal block size based on input dimensions
        2. Allocates output tensor with same shape as input
        3. Launches triton kernel with grid_size = batch_size
        4. Each kernel instance processes one row of the input tensor
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    num_warps = 8 if BLOCK_SIZE >= 512 else 4

    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies standard PyTorch softmax for verification.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, dim)

    Returns:
        torch.Tensor: Output tensor with softmax applied along last dimension
    """
    return torch.nn.functional.softmax(x, dim=-1)

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        return torch.randn(batch_size, dim_size, device=device, dtype=dtype)

    from benchmark import verify_correctness, benchmark_performance

    verify_correctness(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
