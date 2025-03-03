API: deepseek
Test Case: ./func_rmsnorm/
Timestamp: 20250303_003132

<think>
Okay, I need to generate Triton code for RMS normalization, test its correctness, and benchmark its performance. Let's start by understanding what RMSNorm does. From what I remember, RMSNorm normalizes the input by dividing by the root mean square of the input across a certain dimension, then scales it with learned weights. The key steps are calculating the variance, normalizing, then scaling.

First, I'll need to create a Triton kernel. Triton uses blocks and grids similar to CUDA, but with its own syntax. The RMSNorm function will require computing the mean of squares along the feature dimension. I'll need to use reduction operations in Triton for that.

For the Triton kernel, I'll process each row of the input tensor. Each program instance (block) handles a row. We'll load the row into shared memory, compute the squared values, then perform a reduction to get the sum of squares. Using that sum, compute the RMS (add epsilon for numerical stability), then normalize each element and multiply by the weight.

In the Python function that calls the Triton kernel, I'll need to handle the dimensions, set the grid size based on the number of rows (batch size), and launch the kernel with appropriate configurations like block size (BLOCK_SIZE). Also, need to handle different data types, maybe using Triton's autotune for different block sizes.

Next, the PyTorch reference implementation. That's straightforward: compute the variance along the last dimension, normalize, scale with weights. Should match the Triton output.

Testing and benchmarking: The provided code uses verify_correctness and benchmark_performance functions from benchmark.py. The input generator creates random tensors and weights, along with a small epsilon. The test runs across various batch and dimension sizes. I need to make sure the Triton function and Torch function are passed correctly, and the input generator is set up to return the required arguments.

Potential issues: Making sure the Triton reduction is correct. Shared memory usage and synchronization points. Handling different data types (like float16). Also, the epsilon might need to be added in a way that avoids division by zero. Need to check the dimensions for the weight vector—should be the same as the feature dimension.

Let me structure the Triton kernel. The function will take the input tensor, weight tensor, output tensor, stride information, eps, and feature dimension size. The kernel uses multiple stages for reduction. Each thread (or Triton program) loads elements, squares them, then reduces across the columns. Then compute 1/sqrt(variance + eps), and multiply each element by that and the weight.

For the grid in the launch config, the number of blocks should be the batch size, since each block processes a row. The BLOCK_SIZE could be set to the next power of two greater than the feature dimension, but maybe using 1024 as a default, but perhaps using Triton's autotune for different sizes for better performance.

Wait, but in the input generator, the dim_sizes can be up to 4096. So the Triton kernel needs to handle varying feature dimensions. The BLOCK_SIZE should be a power of two that's at least as large as the feature dimension. Or maybe the kernel is designed to handle any size by iterating in steps. For example, if the feature size is larger than the block size, the kernel can loop over chunks.

But in RMSNorm, the feature dimension is the one being normalized over, so the entire row must be processed. So, the Triton kernel needs to handle cases where the feature dimension is larger than the block size. So perhaps the kernel uses a loop, loading elements in chunks of BLOCK_SIZE, accumulating the sum of squares, then doing a reduction.

Alternatively, when launching the kernel, the block size is set to the minimum of the feature dimension and a maximum (like 1024). Wait, but Triton's block size is fixed at compile time. So maybe using a tunable configuration where different block sizes are selected based on the feature dimension. Or using a block size that's a power of two and splitting the processing into multiple steps.

Hmm. This is getting a bit complex. Let's look up an example of Triton RMSNorm implementation. From what I recall, the Triton kernel for layer norm or RMSNorm typically uses a block per row, with the block size being a power of two that divides the feature dimension. For example, if the feature size is 4096, block size 1024 would process 4 elements per thread. So each thread would loop over multiple elements, compute their squares, and then perform a reduction.

In code, the Triton kernel would have a loop over the feature dimension in steps of BLOCK_SIZE. Each thread loads its element, squares it, accumulates the sum. Then after the loop, there's a reduction across the threads in the block to compute the total sum of squares. Then compute the RMS value, and loop again over the feature dimension to apply the normalization and scaling.

So the steps in the Triton kernel are:

1. Compute the row index for the current program instance.
2. Compute the pointers to the input and output rows.
3. Initialize a sum variable to zero.
4. Loop over the columns in steps of BLOCK_SIZE:
   a. Load the input elements.
   b. Square each element and add to the sum.
5. Perform a reduction across all threads in the block to sum all squares.
6. Compute 1 / sqrt(mean + eps), where mean is sum / feature_dim.
7. Loop again over the columns to compute each normalized element multiplied by the weight and the computed RMS value.

Now, in Triton, the reduction can be done using tl.sum with the appropriate axis. Also, using shared memory for the intermediate sums.

Wait, but when the feature dimension is larger than the block size, each thread needs to process multiple elements. For example, if the feature size is 4096 and block size is 512, each thread would handle 8 elements. So the loop would run 8 times.

Putting this together, the Triton kernel would have a loop for the sum accumulation, then the reduction, then another loop for applying the normalization.

Now, for the code:

The Triton function would be decorated with @triton.jit, take the input, weight, output, and other parameters. The function would compute the row index, then calculate the sum of squares, then normalize and scale.

In the Python wrapper function, we'll launch the kernel with grid=(batch_size,), and appropriate block size, which is a power of two. Maybe using 512 as the default block size, but allowing for autotuning.

Then, the Torch function is straightforward: compute the variance along the last dimension, then x * (weight * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()).

Testing with the given input generator, which creates x, weight, and eps. The verify_correctness and benchmark_performance functions are called with the generated Triton and Torch functions.

Potential edge cases: very small batch sizes or feature dimensions, but the given test covers from 64 to 4096 feature sizes, and batch from 256 to 2048.

I need to make sure the Triton kernel handles different sizes, which the loop over the feature dimension in steps of BLOCK_SIZE should handle.

So the code outline is:

- Triton kernel function with the described steps.
- Python function that calls the kernel, handling the grid and parameters.
- Torch reference function.
- The code to call verify and benchmark, using the provided input generator.

Now, let's code this step by step.
</think>

```python
# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr,  # Pointer to input tensor [M, N]
    w_ptr,  # Pointer to weight tensor [N]
    y_ptr,  # Pointer to output tensor [M, N]
    stride_xm,  # Stride between rows of input
    stride_yn,  # Stride between columns of output
    N,  # Size of feature dimension
    eps,  # Numerical stability term
    BLOCK_SIZE: tl.constexpr,  # Triton block size (power of two)
):
    """
    Triton kernel for RMS Normalization across feature dimension.

    Args:
        x_ptr (tensor): Input tensor of shape [M, N]
        w_ptr (tensor): Weight tensor of shape [N]
        y_ptr (tensor): Output tensor of shape [M, N]
        stride_xm (int): Stride between input rows
        stride_yn (int): Stride between output columns
        N (int): Feature dimension size
        eps (float): Small value for numerical stability
        BLOCK_SIZE (tl.constexpr): Triton configuration parameter

    Explanation:
        1. Calculate row index for current block
        2. Compute memory offsets for input/output access
        3. Iteratively compute sum of squares across feature dimension
        4. Use parallel reduction to calculate total squared sum
        5. Compute RMS value with epsilon stabilization
        6. Normalize and scale inputs using weights
    """
    row = tl.program_id(0)
    x_ptrs = x_ptr + row * stride_xm + tl.arange(0, BLOCK_SIZE)
    w_ptrs = w_ptr + tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum and load pointers
    sum_squares = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x = tl.load(x_ptrs + cols * stride_yn, mask=mask, other=0.0)
        x_sq = x * x
        sum_squares += tl.sum(x_sq, axis=0)

    # Calculate RMS value
    mean_squares = sum_squares / N
    rms = 1.0 / tl.sqrt(mean_squares + eps)

    # Apply normalization and weights
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x = tl.load(x_ptrs + cols * stride_yn, mask=mask)
        w = tl.load(w_ptrs + cols, mask=mask)
        y = x * rms * w
        
        tl.store(y_ptr + row * stride_yn + cols * stride_yn, y, mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton implementation of RMS Normalization.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size]
        weight (torch.Tensor): Weight tensor of shape [dim_size]
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized output tensor of shape [batch_size, dim_size]

    Explanation:
        1. Validate input dimensions
        2. Allocate output tensor
        3. Determine optimal block size (nearest power of two <= dim_size)
        4. Launch Triton kernel with calculated grid configuration
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    M, N = x.shape
    y = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096
    
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    
    rmsnorm_kernel[(M,)](
        x, weight, y,
        x.stride(0), y.stride(1),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch reference implementation of RMS Normalization.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size]
        weight (torch.Tensor): Weight tensor of shape [dim_size]
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized output tensor of shape [batch_size, dim_size]

    Explanation:
        1. Compute mean of squared values along last dimension
        2. Add epsilon for numerical stability
        3. Compute reciprocal square root of adjusted mean
        4. Multiply normalized values with weights
    """
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return x * rms * weight

if __name__ == "__main__":
    from benchmark import verify_correctness, benchmark_performance

    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        weight = torch.randn(dim_size, device=device, dtype=dtype)
        eps = 1e-5
        return x, weight, eps
    
    verify_correctness(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
```

<test>
To verify correctness and benchmark performance:

```bash
python rmsnorm_triton.py
```
This will:
1. Validate numerical equivalence between Triton and PyTorch implementations across specified batch/dim sizes
2. Measure execution time and memory usage for both implementations
3. Generate performance comparison charts in ./benchmark_results directory
</test>