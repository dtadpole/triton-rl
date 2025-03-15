API: deepseek
Test Case: ./func_rmsnorm/
Timestamp: 20250315_075251

<think>
Okay, I need to generate Triton code for RMSNorm and test it using the provided benchmark code. Let's start by recalling what RMSNorm is. RMSNorm is a normalization technique where each element is divided by the root mean square of the inputs, scaled by a learned weight. The formula is: output = (input / sqrt(mean(input^2) + epsilon)) * weight.

First, I'll create a PyTorch version of RMSNorm. The torch function should take the input tensor, weight, and epsilon. It calculates the variance along the last dimension, adds epsilon for numerical stability, normalizes the input, and then scales by the weight.

Next, the Triton kernel. Triton requires writing a GPU kernel with careful memory management. For RMSNorm, the input is a 2D tensor (batch_size, dim_size). The kernel should process each row (batch element) by computing the mean of squares, then normalizing each element. 

In Triton, I'll need to define the kernel with appropriate block sizes. Each block can handle a row. We'll load the row into shared memory to compute the sum of squares efficiently. Then, compute the RMS and scale each element. The weight vector is applied element-wise after normalization.

I need to handle different block sizes. Maybe using a heuristic where the block size is the next power of two up to 1024, based on the dimension size. This allows for efficient memory access and parallel computation.

For the Triton function, the steps are:
1. Compute the squared values of the input.
2. Sum the squared values along the last dimension.
3. Compute the mean by dividing the sum by the dimension size.
4. Add epsilon and compute the reciprocal of the square root.
5. Multiply the input by this scale factor and then by the weight.

Testing the correctness involves comparing the Triton output with the PyTorch implementation across various batch and dimension sizes. The benchmark code provided uses verify_correctness_func and benchmark_performance_func, so I need to make sure the Triton and PyTorch functions are correctly passed into these.

Potential issues: Handling different data types (float16), ensuring the Triton kernel correctly synchronizes threads, and managing memory access patterns for efficiency. Also, verifying that the epsilon is added before taking the square root to avoid division by zero.

I should also check if the Triton kernel is compatible with the input generator function, which produces x, weight, and eps. Both functions (Triton and Torch) must accept these three arguments.

Another consideration is the grid parameter in the Triton kernel. The grid should be set to (batch_size, ) so that each row is processed by a separate kernel instance.

Finally, integrating everything into the provided code structure. The main code will import the benchmark functions and run the tests with the specified batch and dimension sizes. The test commands should just run the script, which executes the verification and benchmarking.
</think>

```python
# filename: rmsnorm_triton.py

import torch
import triton
import triton.language as tl

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch implementation of RMS normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
        weight (torch.Tensor): Scale tensor of shape (dim_size,)
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized tensor of same shape as input

    Explanation:
        1. Compute squared values of input
        2. Calculate mean of squares along last dimension
        3. Normalize input using root mean square + epsilon
        4. Scale normalized values with weight vector
    """
    var = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + eps)
    return x_norm * weight

@triton.jit
def _rmsnorm_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    dim_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for RMS normalization.

    Args:
        x_ptr (tl.tensor): Input tensor pointer of shape (batch_size, dim_size)
        weight_ptr (tl.tensor): Weight tensor pointer of shape (dim_size,)
        output_ptr (tl.tensor): Output tensor pointer
        dim_size (tl.constexpr): Dimension size of input
        eps (tl.constexpr): Numerical stability term
        BLOCK_SIZE (tl.constexpr): Block size for parallel computation

    Explanation:
        1. Load input and weight data in blocks
        2. Compute sum of squares for normalization
        3. Calculate normalization scale factor
        4. Apply normalization and scaling
    """
    row_idx = tl.program_id(0)
    offsets = row_idx * dim_size + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < dim_size

    # Load input and weights
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    # Compute sum of squares
    x_square = x * x
    sum_squares = tl.sum(x_square, axis=0) / dim_size

    # Calculate normalization factor
    rms = tl.math.rsqrt(sum_squares + eps)
    
    # Normalize and scale
    output = x * rms * weight
    tl.store(output_ptr + offsets, output, mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton-accelerated RMS normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
        weight (torch.Tensor): Scale tensor of shape (dim_size,)
        eps (float): Numerical stability term

    Returns:
        torch.Tensor: Normalized output tensor of same shape as input

    Explanation:
        1. Validate input dimensions
        2. Determine optimal block size
        3. Launch kernel with appropriate grid configuration
    """
    batch_size, dim_size = x.shape
    assert weight.shape == (dim_size,), "Weight shape mismatch"

    # Determine block size (next power of two up to 1024)
    BLOCK_SIZE = triton.next_power_of_2(dim_size)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024

    output = torch.empty_like(x)
    grid = (batch_size,)
    
    _rmsnorm_kernel[grid](
        x, weight, output,
        dim_size=dim_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

if __name__ == "__main__":
    # Benchmark configurations
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
To verify correctness and benchmark performance, execute the following command:

```bash
python rmsnorm_triton.py
```

This will:
1. Run numerical equivalence tests across different batch/dim sizes
2. Generate performance comparison charts
3. Save results to JSON files (code.correctness.json, code.performance.func.json)
</test>