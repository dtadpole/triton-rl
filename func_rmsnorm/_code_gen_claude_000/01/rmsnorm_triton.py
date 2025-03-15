# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    PyTorch implementation of RMSNorm (Root Mean Square Layer Normalization).

    Args:
        x (torch.Tensor): Input tensor to be normalized. [batch_size, dim_size]
        weight (torch.Tensor): Learned scale parameters. [dim_size]
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Normalized and scaled tensor. [batch_size, dim_size]

    Explanation:
        1. Calculate the root mean square (RMS) along the last dimension.
        2. Normalize the input by dividing by the RMS.
        3. Scale with learned parameters.
    """
    # Calculate the root mean square along the last dimension
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    # Normalize
    y = x / rms
    # Scale with learned parameters
    return y * weight

@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    batch_size, dim_size,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for RMSNorm computation.

    Args:
        x_ptr (tl.tensor): Pointer to input tensor. [batch_size, dim_size]
        weight_ptr (tl.tensor): Pointer to weight tensor. [dim_size]
        output_ptr (tl.tensor): Pointer to output tensor. [batch_size, dim_size]
        batch_size (int): Number of elements in the batch.
        dim_size (int): Size of the dimension to normalize over.
        eps (float): Small constant for numerical stability.
        BLOCK_SIZE (tl.constexpr): Size of blocks to process.

    Explanation:
        1. Get the program ID for batch dimension.
        2. Compute sum of squares for the current batch element.
        3. Calculate the RMS (root mean square).
        4. Load input elements and weights in blocks.
        5. Normalize input by dividing by RMS and multiply by weights.
        6. Store the normalized and scaled results.
    """
    # Program ID for batch dimension
    pid = tl.program_id(0)
    
    # Check if we're within batch bounds
    if pid >= batch_size:
        return
    
    # Compute pointers for this batch
    x_offset = pid * dim_size
    out_offset = pid * dim_size
    
    # Compute sum of squares for this batch
    sum_squares = 0.0
    
    # Process the dimension in blocks
    for block_start in range(0, dim_size, BLOCK_SIZE):
        # Actual block size (might be smaller at the end)
        block_size = min(BLOCK_SIZE, dim_size - block_start)
        
        # Create mask for valid elements
        mask = tl.arange(0, BLOCK_SIZE) < block_size
        
        # Load input block
        x_block_ptr = x_ptr + x_offset + block_start
        x_block = tl.load(x_block_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        
        # Square and sum
        sum_squares += tl.sum(x_block * x_block, axis=0)
    
    # Compute RMS
    rms = tl.sqrt(sum_squares / dim_size + eps)
    
    # Process dimension again to normalize and scale
    for block_start in range(0, dim_size, BLOCK_SIZE):
        # Actual block size
        block_size = min(BLOCK_SIZE, dim_size - block_start)
        
        # Create mask
        mask = tl.arange(0, BLOCK_SIZE) < block_size
        
        # Load input
        x_block_ptr = x_ptr + x_offset + block_start
        x_block = tl.load(x_block_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        
        # Load weight
        weight_block_ptr = weight_ptr + block_start
        weight_block = tl.load(weight_block_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        
        # Normalize and scale
        output_block = (x_block / rms) * weight_block
        
        # Store result
        output_block_ptr = output_ptr + out_offset + block_start
        tl.store(output_block_ptr + tl.arange(0, BLOCK_SIZE), output_block, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=['batch_size', 'dim_size'],
)
def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Triton implementation of RMSNorm (Root Mean Square Layer Normalization).

    Args:
        x (torch.Tensor): Input tensor to be normalized. [batch_size, dim_size]
        weight (torch.Tensor): Learned scale parameters. [dim_size]
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Normalized and scaled tensor. [batch_size, dim_size]

    Explanation:
        1. Get the shape of the input tensor.
        2. Allocate memory for the output tensor.
        3. Determine the optimal block size for processing.
        4. Launch the Triton kernel to perform the normalization.
        5. Return the normalized and scaled tensor.
    """
    batch_size, dim_size = x.shape
    output = torch.empty_like(x)
    
    # Launch kernel
    grid = (batch_size,)
    rmsnorm_kernel[grid](
        x.data_ptr(), weight.data_ptr(), output.data_ptr(),
        batch_size, dim_size,
        eps,
        BLOCK_SIZE=min(1024, triton.next_power_of_2(dim_size))
    )
    
    return output

# Main execution for testing and benchmarking
if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        weight = torch.randn(dim_size, device=device, dtype=dtype)
        eps = 1e-5
        return x, weight, eps
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    
    verify_correctness_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
