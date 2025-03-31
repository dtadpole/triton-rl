Timestamp: 20250331_043059

<think>
Okay, I need to create a Triton module for PyTorch's nn.Linear, implementing both forward and backward passes. Let's start by understanding what's required.

First, the TritonLinear class should mimic the functionality of torch.nn.Linear, so it needs weight as a parameter and optionally bias. Since the user specified to use autotune for block sizes (BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O) in both forward and backward kernels, I need to set up autotuned configurations using Triton's autotune decorator.

For the forward pass kernel:
- The input is a 2D tensor (B, I), and the weight is (O, I). The output is (B, O).
- The kernel should use matrix multiplication. Using Triton's tensor operations, I'll need to handle blocks efficiently.
- Autotune parameters for block sizes and group sizes need to be defined. The configs should vary these block sizes to find the optimal performance.

For the backward pass:
- We need to compute gradients for both the input (dx) and the weight (dw).
- The backward kernel for the input (dx) involves multiplying the gradient of the output (d_output) with the weight matrix.
- The backward kernel for the weight (dw) involves a matrix multiplication between the input and the transpose of d_output.
- Autotune configurations here as well, possibly different from forward.

I should structure each Triton kernel with proper comments as per the template, including description, args, returns, and explanation steps.

Now, implementing the kernels with autotune. For each kernel, I'll use @triton.autotune with a list of configs that specify different block sizes and group sizes. The configs will be combinations of BLOCK_SIZE_* and GROUP_SIZE_* parameters.

In the forward kernel, I'll loop over the input and weight matrices in blocks, accumulating results. Similarly, the backward kernels will handle their respective matrix operations.

The TritonLinear class will have a forward method that calls the Triton kernel. For backward, since PyTorch's autograd is used, I need to create a backward function that computes gradients using Triton kernels and sets them in the parameters.

Wait, but how to integrate the Triton kernels with PyTorch's autograd? I think I need to create a custom autograd Function that encapsulates the forward and backward Triton kernels. So the TritonLinear module will use this custom Function in its forward pass.

Yes, that's right. The module's forward will call an apply method of the custom Function, which handles both forward and backward with Triton.

So, the steps are:

1. Define the forward Triton kernel with autotune.
2. Define the backward kernels (dx and dw) with their own autotune configs.
3. Create a custom autograd Function (TritonLinearFunction) that uses these kernels.
4. The TritonLinear module initializes the weight and optionally bias, and in forward, applies the custom Function.

Wait, but the user's test code doesn't use bias, since in module_generator, the torch.nn.Linear is created with bias=False. So maybe for simplicity, I can omit bias handling in this implementation, or include it but set to None.

Looking at the test code: the TritonLinear is initialized with I and O, and the torch module has bias=False. So in the TritonLinear, we should not include a bias term. So, no bias handling needed here.

Now, for the kernel implementations:

In the forward kernel, the input is (B, I), weight is (O, I), output is (B, O). The kernel will perform input @ weight.T.

But in Triton, how to handle this? The matrix multiplication needs to be broken into blocks. The input and weight are 2D tensors. Each block will compute a portion of the output.

The autotune parameters for the forward kernel would be BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O. But maybe in the kernel, we split the rows and columns into blocks. For example, for the output matrix, each block handles a BLOCK_SIZE_B x BLOCK_SIZE_O tile. Then, for each element in that tile, sum over the K dimension (I) by iterating through blocks of size BLOCK_SIZE_I.

Similarly for the backward kernels.

Another thing to note is that the input is normalized in the test code, so the data is in float16, which Triton handles.

Now, writing the forward kernel:

Using triton.ops.dot for efficient matrix multiplication within blocks. The kernel will have nested loops over the K dimension (I), accumulating the sum.

For the backward kernel for dx (input gradient), the operation is d_output @ weight. So, the kernel will need to multiply d_output (B, O) with weight (O, I), resulting in (B, I).

For the backward kernel for dw (weight gradient), it's input.T @ d_output. So, the input is (B, I), d_output is (B, O), resulting in (I, O). Wait, but the weight's gradient should be of shape (O, I), so maybe we need to transpose input and d_output. Hmm.

Wait, the gradient for the weight matrix is computed as input.T @ d_output. So, if input is (B, I), d_output is (B, O), then input.T is (I, B), multiplied by d_output (B, O) gives (I, O). But the weight is (O, I), so the gradient should be (O, I). Therefore, the weight gradient is (d_output.T @ input). Because input.T @ d_output gives (I, O), and the weight's shape is (O, I), so the gradient is (O, I) which is d_output.T @ input. So, the kernel for dw should perform d_output.T @ input.

So, the dw kernel needs to compute this outer product.

In Triton, implementing these matrix multiplications with tiling and autotuning.

Each kernel will have its own autotune configurations. For example, the forward kernel's configs will have various combinations of block sizes for B, I, O.

But how to structure the autotune parameters for each kernel? Let's see.

In the forward kernel:

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128}, num_warps=4),
        # ... more configs
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def forward_kernel(...):
    ...

Similarly for the backward kernels.

The key parameters are the sizes B, I, O, which vary in the test cases.

Now, implementing the kernels with proper indexing.

For the forward kernel:

Each program instance handles a block in the output matrix. The rows are divided into blocks of BLOCK_SIZE_B, and columns into BLOCK_SIZE_O. For each element in the block, accumulate the sum over I dimension in chunks of BLOCK_SIZE_I.

The pid for the output block is calculated, then the offsets for the input and weight matrices are determined.

Same logic applies for the backward kernels, but the dimensions and matrix orientations are different.

In the backward functions, we need to compute dx and dw. For dx, the operation is d_output @ weight. For dw, it's (d_output.T @ input). So, each requires their own kernel.

Now, the custom autograd Function will have a forward method calling the forward kernel, and a backward method calling the dx and dw kernels.

But integrating the custom Function with the module. The TritonLinear module's forward will call TritonLinearFunction.apply(input, weight), which returns the output. Then, during backward, the Function's backward method will compute the gradients.

But how to handle the weight as a parameter? The custom Function needs to take the weight as an argument, so the apply method will take input and weight.

Wait, in the TritonLinear module, the weight is a Parameter. So, in the forward method of the module, we pass self.weight to the Function.

So, the TritonLinear module's forward is:

def forward(self, x):
    return TritonLinearFunction.apply(x, self.weight)

Then, the backward method of the Function will compute the gradients for x and weight.

Now, writing the Function class:

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # save for backward
        ctx.save_for_backward(x, weight)
        # compute output using Triton kernel
        B, I = x.shape
        O, _ = weight.shape
        output = torch.empty((B, O), device=x.device, dtype=x.dtype)
        # call the forward kernel
        forward_kernel[(triton.cdiv(B, BLOCK_SIZE_B), triton.cdiv(O, BLOCK_SIZE_O))](
            x, weight, output,
            B, I, O,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    @staticmethod
    def backward(ctx, d_output):
        x, weight = ctx.saved_tensors
        # compute dx and dw using Triton kernels
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        B, I = x.shape
        O, _ = weight.shape
        # call dx kernel
        backward_dx_kernel[...](...)
        # call dw kernel
        backward_dw_kernel[...](...)
        return dx, dw

But the actual kernel launches need to be properly configured with the grid and parameters. Also, the autotune configs must be correctly set.

Wait, but in the backward method, we need to compute two gradients: dx (input gradient) and dw (weight gradient). So, two separate kernels are called.

Each backward kernel will have its own autotune configurations.

Now, implementing the dx kernel:

The dx is computed as d_output (B, O) multiplied by weight (O, I), resulting in (B, I). So, each element dx[b, i] = sum over o of d_output[b, o] * weight[o, i].

In Triton, this is another matrix multiplication, but with different dimensions. The kernel for dx will need to tile the input (B, O) and weight (O, I) to compute the (B, I) dx matrix.

Similarly, the dw kernel computes d_output.T (O, B) multiplied by input (B, I), resulting in (O, I) dw. So, each element dw[o, i] = sum over b of d_output[b, o] * input[b, i].

This is another matrix multiplication, but between d_output.T and input.

Each of these operations requires their own Triton kernel with autotuning.

The kernels will have similar structures but different indexing based on the operation.

Another consideration is the grid dimensions for launching the kernels. For example, the dx kernel's grid would be based on B and I, while the dw kernel's grid is based on O and I.

But since the weight gradient is (O, I), the dw kernel will iterate over O and I blocks.

Putting it all together:

The TritonLinear module will have the weight parameter, and the Function will handle the forward and backward passes using the Triton kernels.

Now, testing the code with the provided main block. The main code uses B_sizes, I_sizes, O_sizes to test various configurations. The module_generator creates TritonLinear and torch.nn.Linear modules. The forward and backward generators create input and grad_output tensors.

The verify and benchmark functions from benchmark.py are used to check correctness and measure performance.

Potential issues:

- Ensuring that the Triton kernels are correctly handling the data types and strides.
- Properly setting the autotune configurations for each kernel.
- Handling varying block sizes and group sizes as per autotune.
- Correctness of the matrix multiplications in both forward and backward passes.
- Comparing the results with PyTorch's implementation within acceptable tolerances.

Another thing to note is that the test code uses torch.compile on the modules when benchmarking. But Triton kernels are already compiled, so this might not be necessary. However, the benchmark code calls torch.compile on both the Triton and Torch modules. So the Triton module's forward and backward should be compatible with torch.compile.

Wait, but Triton kernels are JIT compiled when the function is called. When using torch.compile, it's possible that the Triton code is being traced or compiled with TorchInductor. However, in this case, since the TritonLinear uses custom Triton kernels via the autograd Function, torch.compile may not interfere as long as the custom Function is properly handled. But this might require additional testing.

But according to the user's benchmark code in benchmark_performance_forward and benchmark_performance_backward, they compile both modules. So the Triton module must be compatible with torch.compile. However, custom autograd Functions might not be compatible with torch.compile's graph capture. This could be a problem.

Alternatively, maybe the user's benchmark code is incorrect, and compiling the Triton module isn't necessary. Or perhaps the Triton implementation is designed to be compatible.

But for now, perhaps proceed under the assumption that the custom Function is compatible, or that the benchmark code uses the compiled version. Alternatively, maybe the benchmark code is structured to compile the modules but the Triton implementation doesn't require it. However, since the benchmark code is provided, we need to ensure that the TritonLinear can be compiled with torch.compile. That might not be possible if the custom autograd Function uses Triton kernels, which are outside the TorchInductor's scope.

This could be a potential issue. However, the user's test code is given, so perhaps the Triton module is supposed to work without relying on torch.compile. But the benchmark code in benchmark_performance_forward compiles both modules. Hmm. This could lead to incorrect benchmarking if the Triton module is not compatible with torch.compile.

But maybe the TritonLinear module's forward method is written in such a way that it can be compiled by torch.compile. For example, if the custom Function is written using Triton's JIT decorator, and the rest is standard PyTorch code, then torch.compile might handle it by treating the Function as a black box. However, this is speculative.

In any case, proceed with implementing the TritonLinear as per the requirements, using the custom autograd Function with Triton kernels. The benchmark code provided by the user will handle the testing.

Now, putting all the code together.

The filename is linear_triton.py. The code should include the TritonLinear class, the custom Function, and the Triton kernels with autotune.

The kernels need to be properly commented as per the user's template, with Description, Args, Returns, and Explanation sections.

So, the forward kernel's docstring would be:

"""
Performs forward pass of linear layer using matrix multiplication.

Args:
    input_ptr (triton.Pointer): Input tensor of shape (B, I).
    weight_ptr (triton.Pointer): Weight tensor of shape (O, I).
    output_ptr (triton.Pointer): Output tensor of shape (B, O) to be written.
    B (int): Batch size dimension.
    I (int): Input feature dimension.
    O (int): Output feature dimension.
    stride_ib, stride_ii: Strides for input tensor.
    stride_wo, stride_wi: Strides for weight tensor.
    stride_ob, stride_oo: Strides for output tensor.
    BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O: Block sizes for tiling.

Explanation:
    1. Determine the block indices for B and O dimensions.
    2. Iterate over the I dimension in blocks, accumulating partial sums.
    3. Store the accumulated result in the output tensor.
"""

Similarly for the backward kernels.

In code, the autotune decorators will list possible configurations for block sizes. For example, in the forward kernel:

configs = [
    triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
    triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128}, num_warps=8),
    # etc.
]

The key is ['B', 'I', 'O'] as these dimensions vary in the test cases.

For the backward dx kernel, the block sizes might be different:

configs = [
    triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64}, num_warps=4),
    # etc.
]

Similarly for dw kernel.

In the backward method of the Function, after computing dx and dw, return them. Since the forward pass has two inputs (x and weight), the backward returns gradients for both. The bias is not present, so only two gradients are returned.

Putting all this together into code, with proper comments and structure.

Finally, the main block in linear_triton.py should replicate the user's test code exactly, including the module_generator and the calls to verify and benchmark functions from benchmark.py.

Now, testing the code: the user provides test commands in the test section, which should include running the script and generating the correctness and performance reports.
</think>

```python
# filename: linear_triton.py
import torch
import triton
import triton.language as tl
from torch.autograd import Function

class TritonLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O, _ = weight.shape
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        def grid(meta):
            return (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(O, meta['BLOCK_SIZE_O']))
        
        forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O, _ = weight.shape
        
        grad_input = torch.empty_like(input)
        grad_weight = torch.empty_like(weight)

        # Backward for input gradient
        def grid_dx(meta):
            return (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
        
        backward_dx_kernel[grid_dx](
            grad_output, weight, grad_input,
            B, I, O,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1),
        )

        # Backward for weight gradient
        def grid_dw(meta):
            return (triton.cdiv(O, meta['BLOCK_SIZE_O']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
        
        backward_dw_kernel[grid_dw](
            input, grad_output, grad_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
        )

        return grad_input, grad_weight

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 256}, num_warps=16),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_ib, stride_ii,
    stride_wo, stride_wi,
    stride_ob, stride_oo,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
):
    """
    Performs forward pass matrix multiplication for linear layer.

    Args:
        input_ptr (triton.Pointer): Input tensor of shape (B, I).
        weight_ptr (triton.Pointer): Weight tensor of shape (O, I).
        output_ptr (triton.Pointer): Output tensor of shape (B, O) to be written.
        B (int): Batch dimension size.
        I (int): Input feature dimension size.
        O (int): Output feature dimension size.
        stride_ib, stride_ii: Input tensor strides.
        stride_wo, stride_wi: Weight tensor strides.
        stride_ob, stride_oo: Output tensor strides.
        BLOCK_SIZE_B, BLOCK_SIZE_O, BLOCK_SIZE_I: Block sizes for tiling.

    Explanation:
        1. Compute block indices for B and O dimensions.
        2. Initialize accumulator for output block.
        3. Iterate over I dimension in blocks, loading input and weight tiles.
        4. Compute matrix product and accumulate results.
        5. Store final results to output tensor.
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_ib + offs_i[None, :] * stride_ii
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi
    
    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, BLOCK_SIZE_I):
        mask_input = (offs_b[:, None] < B) & (offs_i[None, :] + i < I)
        mask_weight = (offs_o[:, None] < O) & (offs_i[None, :] + i < I)
        
        input_tile = tl.load(input_ptrs + i * stride_ii, mask=mask_input, other=0.0)
        weight_tile = tl.load(weight_ptrs + i * stride_wi, mask=mask_weight, other=0.0)
        
        accumulator += tl.dot(input_tile, weight_tile.T)
    
    offs_b_out = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o_out = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    mask = (offs_b_out[:, None] < B) & (offs_o_out[None, :] < O)
    tl.store(output_ptr + offs_b_out[:, None] * stride_ob + offs_o_out[None, :] * stride_oo, 
             accumulator.to(tl.float16), mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def backward_dx_kernel(
    grad_output_ptr, weight_ptr, grad_input_ptr,
    B, I, O,
    stride_go_b, stride_go_o,
    stride_wo, stride_wi,
    stride_gi_b, stride_gi_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
):
    """
    Computes input gradient (dx) via matrix multiplication of grad_output and weight.

    Args:
        grad_output_ptr (triton.Pointer): Gradient output tensor (B, O).
        weight_ptr (triton.Pointer): Weight tensor (O, I).
        grad_input_ptr (triton.Pointer): Gradient input tensor (B, I) to be written.
        B, I, O: Dimension sizes.
        strides: Tensor strides for all pointers.
        BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O: Tiling block sizes.

    Explanation:
        1. Compute block indices for B and I dimensions.
        2. Load grad_output and weight tiles.
        3. Accumulate matrix product over O dimension.
        4. Store result to grad_input.
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    
    go_ptrs = grad_output_ptr + offs_b[:, None] * stride_go_b + offs_o[None, :] * stride_go_o
    w_ptrs = weight_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi
    
    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for o in range(0, O, BLOCK_SIZE_O):
        mask_go = (offs_b[:, None] < B) & (offs_o[None, :] + o < O)
        mask_w = (offs_o[:, None] + o < O) & (offs_i[None, :] < I)
        
        go_tile = tl.load(go_ptrs + o * stride_go_o, mask=mask_go, other=0.0)
        w_tile = tl.load(w_ptrs + o * stride_wo, mask=mask_w, other=0.0)
        
        accumulator += tl.dot(go_tile, w_tile)
    
    gi_ptrs = grad_input_ptr + offs_b[:, None] * stride_gi_b + offs_i[None, :] * stride_gi_i
    mask = (offs_b[:, None] < B) & (offs_i[None, :] < I)
    tl.store(gi_ptrs, accumulator.to(tl.float16), mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def backward_dw_kernel(
    input_ptr, grad_output_ptr, grad_weight_ptr,
    B, I, O,
    stride_ib, stride_ii,
    stride_go_b, stride_go_o,
    stride_gw_o, stride_gw_i,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
):
    """
    Computes weight gradient (dw) via matrix multiplication of input and grad_output.

    Args:
        input_ptr (triton.Pointer): Input tensor (B, I).
        grad_output_ptr (triton.Pointer): Gradient output tensor (B, O).
        grad_weight_ptr (triton.Pointer): Weight gradient tensor (O, I) to be written.
        B, I, O: Dimension sizes.
        strides: Tensor strides for all pointers.
        BLOCK_SIZE_O, BLOCK_SIZE_I, BLOCK_SIZE_B: Tiling block sizes.

    Explanation:
        1. Compute block indices for O and I dimensions.
        2. Load input and grad_output tiles.
        3. Accumulate outer product over B dimension.
        4. Store result to grad_weight.
    """
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = tl.arange(0, BLOCK_SIZE_B)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_ib + offs_i[None, :] * stride_ii
    go_ptrs = grad_output_ptr + offs_b[:, None] * stride_go_b + offs_o[None, :] * stride_go_o
    
    accumulator = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    for b in range(0, B, BLOCK_SIZE_B):
        mask_input = (offs_b[:, None] + b < B) & (offs_i[None, :] < I)
        mask_go = (offs_b[:, None] + b < B) & (offs_o[None, :] < O)
        
        input_tile = tl.load(input_ptrs + b * stride_ib, mask=mask_input, other=0.0)
        go_tile = tl.load(go_ptrs + b * stride_go_b, mask=mask_go, other=0.0)
        
        accumulator += tl.dot(go_tile.T, input_tile)
    
    gw_ptrs = grad_weight_ptr + offs_o[:, None] * stride_gw_o + offs_i[None, :] * stride_gw_i
    mask = (offs_o[:, None] < O) & (offs_i[None, :] < I)
    tl.store(gw_ptrs, accumulator.to(tl.float16), mask=mask)

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        torch.nn.init.xavier_normal_(self.weight)
        
    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

if __name__ == "__main__":
    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    def module_generator(B, I, O, device='cuda', dtype=torch.float16):
        triton_module = TritonLinear(I, O, device=device, dtype=dtype)
        torch_module = torch.nn.Linear(I, O, bias=False, device=device, dtype=dtype)
        return triton_module, torch_module

    def forward_generator(B, I, O, device='cuda', dtype=torch.float16):
        input = torch.randn(B, I, device=device, dtype=dtype)
        input = torch.nn.functional.normalize(input, dim=-1)
        return input

    from benchmark import verify_correctness_forward, benchmark_performance_forward
    verify_correctness_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes])
    benchmark_performance_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes])

    def backward_generator(B, I, O, device='cuda', dtype=torch.float16):
        input = torch.randn(B, I, device=device, dtype=dtype)
        input = torch.nn.functional.normalize(input, dim=-1)
        d_output = torch.randn(B, O, device=device, dtype=dtype)
        d_output = torch.nn.functional.normalize(d_output, dim=-1)
        return input, d_output

    from benchmark import verify_correctness_backward, benchmark_performance_backward
    verify_correctness_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes])
    benchmark_performance_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes])
```

<test>
To test the implementation, run the following command:

```bash
python linear_triton.py
```

This will execute the correctness verification and performance benchmarking for both forward and backward passes across different tensor sizes. The results will be saved in JSON files, and performance plots will be generated.
</test>