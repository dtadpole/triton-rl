# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    stride_b, stride_h, stride_w,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight
    
    Args:
        x_ptr (tl.tensor): Pointer to input tensor of shape [n_rows, n_cols]
        weight_ptr (tl.tensor): Pointer to weight tensor of shape [n_cols]
        output_ptr (tl.tensor): Pointer to output tensor of shape [n_rows, n_cols]
        stride_b (int): Stride for batch dimension of input
        stride_h (int): Stride for column dimension of input
        stride_w (int): Stride for column dimension of weight
        n_rows (int): Number of rows in the input tensor
        n_cols (int): Number of columns in the input tensor
        eps (float): Small constant for numerical stability
        BLOCK_SIZE (tl.constexpr): Size of tile in the inner dimension

    Explanation:
        1. Each program processes one row of the input tensor.
        2. Compute the sum of squares for the entire row in tiles.
        3. Calculate the RMS normalization factor as 1/sqrt(mean(x^2) + eps).
        4. Apply normalization and weight to each element in the row.
        5. Store the results in the output tensor.
    """
    # Get the program ID - each program handles one row
    pid = tl.program_id(0)
    
    # Only process within the number of rows
    if pid < n_rows:
        # Compute row offset for this program
        row_start_ptr = x_ptr + pid * stride_b
        out_row_start_ptr = output_ptr + pid * stride_b
        
        # Initialize accumulator for variance
        var = 0.0
        
        # Calculate variance for the row
        for i in range(0, n_cols, BLOCK_SIZE):
            # Create block offsets and masks
            cols = i + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            
            # Load the input block
            x_ptrs = row_start_ptr + cols * stride_h
            x = tl.load(x_ptrs, mask=mask, other=0.0)
            
            # Accumulate squared values
            var += tl.sum(x * x * mask)
        
        # Calculate RMS = 1/sqrt(mean(x^2) + eps)
        rms = tl.math.rsqrt(var / n_cols + eps)
        
        # Apply normalization and weight
        for i in range(0, n_cols, BLOCK_SIZE):
            # Create block offsets and masks
            cols = i + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            
            # Load the input block and weights
            x_ptrs = row_start_ptr + cols * stride_h
            x = tl.load(x_ptrs, mask=mask, other=0.0)
            
            weight_ptrs = weight_ptr + cols * stride_w
            weight = tl.load(weight_ptrs, mask=mask, other=1.0)
            
            # Calculate output
            output = x * rms * weight
            
            # Store the result
            out_ptrs = out_row_start_ptr + cols * stride_h
            tl.store(out_ptrs, output, mask=mask)

def rmsnorm_triton(x, weight, eps=1e-5):
    """
    Apply RMSNorm to the input tensor using Triton.
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size]
        weight (torch.Tensor): Weight tensor of shape [dim_size]
        eps (float): Small constant for numerical stability
    
    Returns:
        torch.Tensor: Normalized tensor of shape [batch_size, dim_size]
    
    Explanation:
        1. Extract tensor dimensions.
        2. Allocate an output tensor with the same shape and dtype as the input.
        3. Launch a Triton kernel with one program per row of the input tensor.
        4. Each program computes the RMSNorm for its assigned row.
        5. Return the normalized output tensor.
    """
    # Extract tensor dimensions
    batch_size, dim_size = x.shape
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Launch kernel with one thread per row
    grid = (batch_size,)
    rmsnorm_kernel[grid](
        x.data_ptr(), weight.data_ptr(), output.data_ptr(),
        x.stride(0), x.stride(1), weight.stride(0),
        batch_size, dim_size,
        eps,
        BLOCK_SIZE=min(triton.next_power_of_2(dim_size), 1024),
    )
    
    return output

def rmsnorm_torch(x, weight, eps=1e-5):
    """
    PyTorch implementation of RMSNorm.
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size]
        weight (torch.Tensor): Weight tensor of shape [dim_size]
        eps (float): Small constant for numerical stability
    
    Returns:
        torch.Tensor: Normalized tensor of shape [batch_size, dim_size]
    
    Explanation:
        1. Compute the mean of squared values along the last dimension.
        2. Normalize the input by dividing by the root mean square (RMS).
        3. Scale the normalized values by the weight parameter.
        4. Return the result.
    """
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight

# Test correctness and benchmark performance
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
