API: deepseek
Test Case: ./func_rmsnorm/
Timestamp: 20250315_075229

<think>
Okay, I need to generate Triton code for RMSNorm and set up testing and benchmarking using the provided code. Let's start by understanding what RMSNorm does. RMSNorm normalizes the input by the root mean square of the features, then scales it using learned weights. The key steps are computing the variance, normalizing, and scaling.

First, I'll create the Triton kernel. Triton requires handling memory efficiently. The input is a 2D tensor (batch_size, dim_size). The kernel should compute the squared sum across the feature dimension, apply normalization, and multiply by the weight.

For the Triton function, I'll need to use block-wise parallelism. Each block can process a row (batch element). I'll load the row into shared memory, compute the sum of squares, then normalize each element. The weight vector is applied element-wise after normalization.

Next, the PyTorch reference implementation. It's straightforward using tensor operations: calculate the variance, add epsilon for numerical stability, normalize, and multiply by weight.

The input generator function creates random tensors for input and weight, along with a small epsilon. This matches the parameters needed for both Triton and PyTorch functions.

Testing involves calling the verify_correctness_func and benchmark_performance_func from benchmark.py. The given code passes the batch and dimension sizes, so the generated code should import these functions and execute them with the appropriate arguments.

I need to ensure that the Triton kernel is properly configured with grid dimensions. The grid should have a size equal to the batch size, processing each row in parallel. The BLOCK_SIZE is a tunable parameter, probably set to 1024, which is common for Triton kernels.

Potential issues: Handling different dimension sizes that aren't multiples of BLOCK_SIZE. Triton automatically handles this with masking. Also, ensuring that the weight vector is properly broadcasted across the batch.

Now, structuring the code. The Triton function will be in rmsnorm_triton.py. It includes the kernel and a wrapper function. The PyTorch function is in the same file. The main block calls the benchmark functions with the input generator and configurations.

Testing commands will run the script and check for correctness and performance. The bash command uses Python with the CUDA_VISIBLE_DEVICES set, but that's optional depending on the environment.

I should also add the detailed comments as per the template. Each function (Triton and PyTorch) needs a docstring explaining their purpose, arguments, returns, and steps.

Double-checking the Triton kernel's memory operations: loading the input row, computing sum of squares, normalizing, then applying weights. Using tl.where to handle masking for non-divisible block sizes.

Finally, the test section provides instructions to run the script and generate the correctness and performance reports.
</think>

```python
# filename: rmsnorm_triton.py

import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for RMS Normalization.

    Args:
        x_ptr (tl.tensor): Pointer to input tensor of shape (B, N)
        weight_ptr (tl.tensor): Pointer to weight tensor of shape (N,)
        output_ptr (tl.tensor): Pointer to output tensor of shape (B, N)
        N (int): Feature dimension size
        eps (float): Small value for numerical stability
        BLOCK_SIZE (tl.constexpr): Triton block size

    Explanation:
        1. Each program processes one row of the input matrix
        2. Load data from global memory into SRAM
        3. Compute sum of squares for normalization
        4. Apply normalization and weight scaling
        5. Write results back to global memory
    """
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Load input row with boundary check
    x_ptrs = x_ptr + row_idx * N + cols
    mask = cols < N
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Compute variance
    x_float = tl.math.astype(x, tl.float32)
    sum_sq = tl.sum(x_float * x_float, axis=0) / N
    rms = tl.math.rsqrt(sum_sq + eps)
    
    # Normalize and scale
    normalized = x_float * rms
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    output = normalized * tl.math.astype(weight, tl.float32)
    
    # Write back results
    tl.store(x_ptrs, tl.math.astype(output, x.dtype), mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton implementation of RMSNorm.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
        weight (torch.Tensor): Weight tensor of shape (dim_size,)
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized output tensor of same shape as input

    Explanation:
        1. Validate input dimensions
        2. Allocate output tensor
        3. Configure kernel launch parameters
        4. Launch RMSNorm kernel
    """
    B, N = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096
        
    grid = (B,)
    rmsnorm_kernel[grid](
        x, weight, output,
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch reference implementation of RMSNorm.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
        weight (torch.Tensor): Weight tensor of shape (dim_size,)
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized output tensor of same shape as input

    Explanation:
        1. Compute root mean square along feature dimension
        2. Normalize input using computed RMS
        3. Apply learned weight scaling
    """
    rms = torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps)
    return (x * rms).to(x.dtype) * weight

if __name__ == "__main__":
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
To test the RMSNorm implementation:

1. Save the code as `rmsnorm_triton.py`
2. Ensure `benchmark.py` exists in the same directory
3. Run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python rmsnorm_triton.py
```

This will:
- Generate correctness reports in `code.correctness.json`
- Create performance reports in `code.performance.func.json`
- Produce visualization plots for performance comparison
- Print median timing results to stdout
</test>