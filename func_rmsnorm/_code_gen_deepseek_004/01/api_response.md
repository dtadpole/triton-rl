API: deepseek
Test Case: ./func_rmsnorm/
Timestamp: 20250315_075231

<think>
Okay, I need to generate Triton code for RMSNorm, along with a PyTorch reference implementation. Then, test correctness and benchmark performance using the provided code. Let's start by understanding what RMSNorm does.

RMSNorm is a normalization technique where the input is scaled by the root mean square of the features. The formula is: output = (input / sqrt(mean(input^2) + eps)) * weight. So, the steps are compute the squared values, take the mean along the feature dimension, add epsilon, take the square root, divide the input by this value, and multiply by the weight.

First, the Triton kernel. Triton requires writing a GPU kernel with certain optimizations. For RMSNorm, the kernel should process the input tensor in blocks. Each block handles a part of the input. Let's think about the dimensions. The input is 2D: batch_size x dim_size. Each program instance (block) can handle a row (batch element). So, the kernel will process each row independently.

In Triton, I need to load the data from the input tensor, compute the squared sum, then normalize. Using Triton's pointer arithmetic, I can iterate over the elements in the row. To optimize, use multiple memory operations and vectorized loads/stores where possible. Also, the weight vector is of size dim_size, so each element in the row is multiplied by the corresponding weight element.

For the PyTorch reference, it's straightforward: compute the variance, normalize, scale by weight. The function should handle 2D inputs.

Next, the test and benchmark code. The given code uses input_generator to create random inputs. The verify_correctness_func compares the Triton and PyTorch outputs. The benchmark_performance_func measures the speed.

Now, setting up the files. The main file (rmsnorm_triton.py) should include both the Triton kernel and the PyTorch function. The Triton kernel uses @triton.jit decorator. The launch configuration (grid) should have a size equal to the batch size, since each row is processed independently.

Potential issues: Handling the eps correctly, ensuring the Triton kernel correctly computes the mean of squares along the feature dimension. Also, the weight is applied element-wise, so each element in the normalized input is multiplied by the corresponding weight element.

Another consideration is the block size (BLOCK_SIZE). Choosing a size that's a power of two and fits into the GPU's constraints. For example, 1024 might be a good starting point, but it's necessary to check if Triton allows that.

Testing the code with varying batch and dimension sizes. The benchmark.py code will handle generating different configurations and comparing results. The input generator creates x and weight with appropriate sizes, and eps is fixed.

Need to ensure that the Triton function and PyTorch function are compatible with the input generator's outputs. The Triton function should accept x, weight, eps as inputs and return the normalized tensor. The PyTorch function does the same.

Including detailed comments as per the template. Each function (Triton and PyTorch) needs a docstring with description, arguments, returns, and step-by-step explanation.

Possible edge cases: very small eps, large dimensions that aren't multiples of the block size. Triton's kernel should handle these by checking if the index is within bounds.

Once the code is written, the test command runs the script. The test section should include instructions to run the Python file which calls the benchmark functions.
</think>

```python
# filename: rmsnorm_triton.py

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,  # Pointer to input tensor, shape (B, D)
    w_ptr,  # Pointer to weight tensor, shape (D,)
    y_ptr,  # Pointer to output tensor, shape (B, D)
    eps,    # Epsilon for numerical stability
    B,      # Batch size
    D,      # Feature dimension size
    stride_x_batch,  # Stride between batch elements in x
    stride_x_dim,    # Stride between features in x
    stride_w_dim,    # Stride between features in weight (typically 1)
    stride_y_batch,  # Stride between batch elements in y
    stride_y_dim,    # Stride between features in y
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute RMS normalization for a batch of vectors using Triton.

    Args:
        x_ptr (triton.Pointer): Input tensor pointer [B, D]
        w_ptr (triton.Pointer): Weight tensor pointer [D]
        y_ptr (triton.Pointer): Output tensor pointer [B, D]
        eps (float): Small value to avoid division by zero
        B (int): Batch size dimension
        D (int): Feature dimension size
        stride_x_batch (int): Stride between x batch elements
        stride_x_dim (int): Stride between x features
        stride_w_dim (int): Stride between weight features
        stride_y_batch (int): Stride between y batch elements
        stride_y_dim (int): Stride between y features
        BLOCK_SIZE (tl.constexpr): Block size for Triton operations
    """
    # Get current program's batch index
    batch_idx = tl.program_id(0)
    
    # Calculate pointers to current batch in input and output
    x_batch_ptr = x_ptr + batch_idx * stride_x_batch
    y_batch_ptr = y_ptr + batch_idx * stride_y_batch
    
    # Initialize sum of squares
    sum_sq = 0.0
    
    # Compute sum of squares in blocks
    for offset in range(0, D, BLOCK_SIZE):
        col = offset + tl.arange(0, BLOCK_SIZE)
        mask = col < D
        
        # Load input values and compute squares
        x = tl.load(x_batch_ptr + col * stride_x_dim, mask=mask, other=0.0)
        x_sq = x * x
        sum_sq += tl.sum(x_sq, axis=0)
    
    # Compute RMS value
    rms = tl.sqrt(sum_sq / D + eps)
    
    # Normalize and scale with weight
    for offset in range(0, D, BLOCK_SIZE):
        col = offset + tl.arange(0, BLOCK_SIZE)
        mask = col < D
        
        # Load input and weight
        x = tl.load(x_batch_ptr + col * stride_x_dim, mask=mask, other=0.0)
        w = tl.load(w_ptr + col * stride_w_dim, mask=mask, other=0.0)
        
        # Compute normalized output
        y = (x / rms) * w
        
        # Store result
        tl.store(y_batch_ptr + col * stride_y_dim, y, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton implementation of RMS normalization.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size]
        weight (torch.Tensor): Weight tensor of shape [dim_size]
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Output tensor of same shape as input

    Explanation:
        1. Verify input dimensions and contiguity
        2. Allocate output tensor
        3. Determine optimal block size (up to 1024)
        4. Launch Triton kernel with appropriate grid configuration
        5. Return normalized and scaled output
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    B, D = x.shape
    assert weight.shape == (D,), "Weight shape mismatch"
    
    # Ensure contiguous memory layout
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = triton.next_power_of_2(D)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    
    grid = (B,)
    
    # Launch kernel
    rmsnorm_kernel[grid](
        x, weight, y, eps,
        B, D,
        x.stride(0), x.stride(1),
        weight.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch reference implementation of RMS normalization.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size]
        weight (torch.Tensor): Weight tensor of shape [dim_size]
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Output tensor of same shape as input

    Explanation:
        1. Compute mean of squared values along feature dimension
        2. Add epsilon and compute reciprocal square root
        3. Normalize input values
        4. Scale with weight vector
    """
    rms = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    return x * (weight / (rms + eps))


if __name__ == "__main__":
    # Configuration parameters from problem statement
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        weight = torch.randn(dim_size, device=device, dtype=dtype)
        eps = 1e-5
        return x, weight, eps
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    
    # Verify numerical correctness
    verify_correctness_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
    
    # Benchmark performance
    benchmark_performance_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
```

<test>
To test the RMSNorm implementation, execute the following command. Ensure you have compatible CUDA hardware and Triton installed.

```bash
python rmsnorm_triton.py
```

This will:
1. Verify numerical equivalence between Triton and PyTorch implementations across specified batch/dimension sizes
2. Benchmark performance and generate comparison results
3. Output correctness reports (code.correctness*.json) and performance plots
</test>