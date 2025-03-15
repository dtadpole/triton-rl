API: deepseek
Test Case: ./module_linear/
Timestamp: 20250314_093632

<think>
Okay, I need to create a Triton module for PyTorch's nn.Linear, implementing both forward and backward passes. The filename should be triton_linear.py. Let me start by understanding what's required here.

First, the TritonLinear class needs to mimic the functionality of torch.nn.Linear. That means in the forward pass, it performs a matrix multiplication between the input and the weight matrix, without bias since the user's code specifies bias=False. The backward pass will compute gradients for both the input and the weight.

I remember that Triton uses kernels which require grid and block configurations. Autotuning is needed for block sizes BLOCK_SIZE_B, BLOCK_SIZE_I, and BLOCK_SIZE_O. So I should use Triton's autotune configs, probably using the @triton.autotune decorator with specific configs for each kernel.

For the forward kernel, the input is of shape (B, I), the weight is (O, I). The output should be (B, O). The matrix multiplication here is input @ weight.T. So in the kernel, each block might handle a block of B, I, O dimensions. The autotune parameters will determine the optimal block sizes for each dimension.

The backward pass requires two kernels: one for computing the gradient of the input (which involves d_output @ weight) and another for the gradient of the weight (which involves input.T @ d_output). These are the standard gradients for a linear layer without bias.

Wait, in the backward function, when computing d_input, the operation is d_output multiplied by the weight matrix. So the kernel for d_input would handle that. For d_weight, since it's a gradient, we need to compute the outer product of input and d_output, summed over the batch dimension. That might be a bit tricky in Triton, as it's a reduction across batches.

Let me structure the kernels:

Forward Kernel:
- Input: (B, I), Weight: (O, I)
- Output: (B, O)
- Each element in the output is a dot product of a row in input and a row in weight (since weight is stored as (O, I), so we need to transpose it for the matmul).

Backward Kernel for d_input:
- d_output (B, O) multiplied by weight (O, I) gives d_input (B, I). So the operation is d_output @ weight.

Backward Kernel for d_weight:
- The gradient for weight is input.T @ d_output. So for each element in the weight gradient (O, I), it's the sum over all batches of input[b, i] * d_output[b, o]. This requires a reduction across the B dimension, which can be handled using atomic additions in Triton.

Now, considering the autotuning. For each kernel (forward, backward_input, backward_weight), I'll need to define different configs for block sizes. The autotune decorator will test these configs and select the fastest one.

Looking at the parameters for the forward kernel: the input is of size B x I, weight is O x I. The output is B x O. So the block sizes for B, I, O need to be tuned. The kernel might split the computation into blocks along these dimensions.

For the backward_input kernel, the operation is d_output (B, O) @ weight (O, I), resulting in d_input (B, I). Block sizes for B, O, I could be relevant here.

For backward_weight, input (B, I) and d_output (B, O) are multiplied, then summed over B to get (I, O) (but transposed to O, I?). Wait, the weight gradient is (I, O), but since weight is (O, I), maybe the gradient is (O, I). Hmm, need to verify the math.

The gradient for weight in a linear layer (without bias) is input.T @ d_output, which is (I, B) @ (B, O) = (I, O). But the weight matrix is (O, I), so the gradient should be (I, O), but transposed. Wait, no. Let's clarify:

In forward pass: output = input @ weight.T (since weight is (O, I), input is (B, I), so weight.T is (I, O), and input @ weight.T is (B, O)).

The gradient for weight would be the derivative of the loss with respect to weight. Let's compute that. The loss gradient dL/dW is equal to (dL/doutput).T @ input, which is (O, B) @ (B, I) = (O, I). So the gradient for weight is (O, I), matching the weight's shape. Therefore, in the backward_weight kernel, we need to compute this product.

So the kernel for d_weight would need to accumulate the sum over B of d_output[b, o] * input[b, i] for each (o, i). This requires a reduction over the batch dimension. Triton can handle this with atomic operations, allowing each block to compute a portion of the batch and accumulate into the gradient tensor.

Now, implementing these kernels with Triton:

Each kernel will have a grid that's determined by the problem size divided by the block size. For example, in the forward pass:

grid = (triton.cdiv(B, BLOCK_SIZE_B), triton.cdiv(O, BLOCK_SIZE_O))

Each block would process a block of B and O dimensions. The K dimension (I) is the reduction dimension, so we need to loop over blocks of I and accumulate the sum.

Autotuning configs for each kernel need to try different combinations of block sizes. For example, for the forward kernel, possible combinations could be BLOCK_SIZE_B=16, 32, 64; BLOCK_SIZE_O=16, 32, 64; BLOCK_SIZE_I=32, 64, etc.

In the TritonLinear class, the parameters are initialized similar to PyTorch's Linear. The forward method will call the forward kernel. For the backward pass, we need to define a backward function that computes the gradients. However, in PyTorch, when using custom autograd Functions, we need to create a Function subclass that implements forward and backward.

Wait, but the user wants to implement TritonLinear as a PyTorch module. So the TritonLinear should subclass nn.Module and manage its own parameters. The forward pass would use a custom autograd Function that wraps the Triton kernels for both forward and backward.

So the TritonLinear module's forward method would call an autograd Function (e.g., TritonLinearFunction) which has its own forward and backward methods. In the forward method of the Function, the Triton kernel is called. In the backward method, the gradients are computed using the Triton backward kernels.

So the structure would be:

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, ...):
        super().__init__()
        self.weight = nn.Parameter(...)
        # initialize weight

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

Then, TritonLinearFunction is a subclass of torch.autograd.Function, which in its forward calls the forward kernel, and in the backward calls the backward kernels for input and weight gradients.

But how to handle the backward pass in Triton? The backward pass for the Function would receive the upstream gradient (d_output) and compute the gradients for the input and weight.

So the TritonLinearFunction's backward method would compute:

- d_input: d_output @ weight (using the backward_input kernel)
- d_weight: input.T @ d_output (using the backward_weight kernel)

Wait, no. The gradient for the input is d_output @ weight. The gradient for the weight is input.T @ d_output (transpose to match the weight's shape). But the weight is (O, I), so the gradient should be (O, I). input is (B, I), d_output is (B, O). So input.T is (I, B), and d_output is (B, O). So input.T @ d_output is (I, O), but we need (O, I). So we need to transpose the result.

Wait, the correct gradient for weight is (d_output.T @ input).T? No, let's rederive:

The forward pass is output = input @ weight.T. The gradient of the loss with respect to weight is the input matrix multiplied by the gradient of the loss with respect to output. Let me compute:

dL/dW = (dL/doutput) @ (doutput/dW). The output is input @ W.T, so the derivative of output with respect to W is a 4D tensor, but matrix calculus gives that the gradient for W is input.T @ d_output.

Wait, perhaps using matrix calculus:

Let Y = X W^T, so dY/dW is such that the gradient dL/dW is X^T @ dL/dY. Because when you have Y = X W^T, then the gradient dL/dW would be the derivative of the loss with respect to W, which is (dL/dY) @ X. Wait, maybe I should double-check.

Using the chain rule: Suppose L is a scalar loss. The gradient dL/dW is computed as the sum over all elements of (dL/dY) * (dY/dW). 

The element Y_{b,o} = sum_i X_{b,i} W_{o,i}.

The derivative of Y_{b,o} with respect to W_{o',i'} is X_{b,i'} if o' = o, else 0. 

So, dL/dW_{o,i} = sum_b (dL/dY_{b,o}) * X_{b,i}

Which is equivalent to (dL/dY)^T @ X. But dL/dY is (B, O), so (dL/dY)^T is (O, B), multiplied by X (B, I) gives (O, I). So the gradient for W is (O, I), which matches the shape of W.

Therefore, the gradient for the weight is (d_output.T @ input). So to compute this in Triton, we need to multiply the transpose of d_output (O, B) with input (B, I), resulting in (O, I), which is the correct shape.

This means that the backward kernel for the weight gradient will perform this matrix multiplication. Since the weight gradient is a sum over batches, each element (o, i) in the gradient is the sum over all b of d_output[b, o] * input[b, i].

Implementing this in Triton requires a kernel that efficiently accumulates these sums. Since the batch dimension can be large, it's efficient to split the computation into blocks and use atomic operations to accumulate the sum.

Now, for the Triton kernels:

Each kernel should have proper indexing to handle the matrix dimensions. For the forward kernel:

- Each program instance (block) handles a block of B and O.
- It loops over blocks of I to compute the sum for each element in the output.

For the backward_input kernel:

- Each block handles a block of B and I.
- It uses the weight matrix and d_output to compute d_input.

For the backward_weight kernel:

- Each block handles a block of O and I.
- It loops over blocks of B to accumulate the sum of d_output[b, o] * input[b, i].

Now, setting up the autotuning for each kernel.

For the forward kernel, we can define autotune configurations with different combinations of BLOCK_SIZE_B, BLOCK_SIZE_O, and BLOCK_SIZE_I. The same applies to the other kernels.

In the code, each kernel will be decorated with @triton.autotune, providing a list of configs with different block sizes. Triton will benchmark these configurations and select the fastest one.

Now, handling the data types. The user specifies that the module should use dtype=torch.float16. So the Triton kernels should be configured with the appropriate data type, using tl.float16.

Also, in the test code provided by the user, the module_generator returns TritonLinear and torch.nn.Linear with the same parameters (bias=False). So the TritonLinear should not include a bias term.

Putting this all together:

The TritonLinear class initializes the weight parameter, and in the forward pass applies the TritonLinearFunction. The function uses the forward Triton kernel. The backward method of the function uses the backward_input and backward_weight kernels.

Potential issues to consider:

- Correctness of the matrix multiplications, ensuring the dimensions are handled properly.
- Efficient memory access patterns in the Triton kernels to maximize performance.
- Atomic operations for accumulating gradients in the backward_weight kernel, especially when multiple threads are updating the same gradient element.

Testing the code with the provided benchmark functions will verify correctness and performance against the PyTorch implementation.

I need to implement each kernel with proper indexing and autotuning. Let's start writing the code step by step.

First, import necessary modules:

import torch
import torch.nn as nn
import triton
import triton.language as tl

Next, define the forward kernel with autotune.

Then the backward kernels.

Then the TritonLinearFunction, which uses these kernels in forward and backward.

Then the TritonLinear module.

I need to make sure that the weight is stored in the correct data type and device. The user's test code uses device='cuda' and dtype=torch.float16, so the TritonLinear should initialize the weight accordingly.

In the forward function of TritonLinearFunction, we call the forward kernel. The input and weight are passed as arguments, and the output is computed.

In backward, the function receives grad_output, computes grad_input and grad_weight using the respective kernels.

Now, writing the kernels:

The forward kernel:

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64}, num_warps=8),
        # More configs can be added here
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr
):
    # ... kernel code ...

The kernel would iterate over the I dimension in blocks, compute the sum for each output element.

Similarly, the backward_input and backward_weight kernels would have their own autotune configs.

Implementing the backward_weight kernel is tricky due to the reduction over B. Each thread block might handle a portion of B and accumulate the sum for a specific O and I.

After defining the kernels, the TritonLinearFunction's forward and backward methods call these kernels.

In the TritonLinear module, the weight is initialized with Kaiming normal initialization, similar to PyTorch's Linear layer.

Testing this code with the provided main block will check correctness and benchmark performance.

Potential edge cases: very small or large B, I, O sizes. The autotune should handle different sizes by selecting appropriate block sizes.

I think that's the general approach. Now, putting all this into code with proper comments as per the user's instruction.
</think>

```python
# filename: triton_linear.py
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128}, num_warps=4),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr
):
    """
    Performs forward matrix multiplication of input and weight matrices.

    Args:
        input_ptr (tensor): Pointer to input tensor of shape (B, I)
        weight_ptr (tensor): Pointer to weight tensor of shape (O, I)
        output_ptr (tensor): Pointer to output tensor of shape (B, O)
        B, I, O (int): Dimensions of input and output
        stride_*: Memory stride information for tensor access
        BLOCK_SIZE_B, BLOCK_SIZE_O, BLOCK_SIZE_I (tl.constexpr): Block sizes for autotuning

    Explanation:
        1. Compute block indices for batch and output dimensions
        2. Initialize output block accumulator
        3. Loop over input dimension blocks and accumulate matrix products
        4. Store final results to output tensor
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, BLOCK_SIZE_I):
        mask_input = (offs_b[:, None] < B) & (offs_i[None, :] + i < I)
        mask_weight = (offs_o[:, None] < O) & (offs_i[None, :] + i < I)
        
        input_block = tl.load(input_ptrs, mask=mask_input, other=0.0)
        weight_block = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(input_block, weight_block, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_I * stride_input_i
        weight_ptrs += BLOCK_SIZE_I * stride_weight_i
    
    offs_b_out = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o_out = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    output_ptrs = output_ptr + offs_b_out[:, None] * stride_output_b + offs_o_out[None, :] * stride_output_o
    mask = (offs_b_out[:, None] < B) & (offs_o_out[None, :] < O)
    tl.store(output_ptrs, acc, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def backward_input_kernel(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, I, O,
    stride_dout_b, stride_dout_o,
    stride_weight_o, stride_weight_i,
    stride_din_b, stride_din_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr
):
    """
    Computes gradient of input for linear layer.

    Args:
        d_output_ptr (tensor): Pointer to output gradient tensor (B, O)
        weight_ptr (tensor): Pointer to weight tensor (O, I)
        d_input_ptr (tensor): Pointer to input gradient tensor (B, I)
        B, I, O (int): Dimension sizes
        stride_*: Memory stride information
        BLOCK_SIZE_B, BLOCK_SIZE_I (tl.constexpr): Block sizes for autotuning

    Explanation:
        1. Calculate block indices for batch and input dimensions
        2. Initialize accumulator
        3. Loop over output dimension blocks and accumulate products
        4. Store final gradient values
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = tl.arange(0, 128)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for o in range(0, O, 128):
        dout_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + (o + offs_o)[None, :] * stride_dout_o
        weight_ptrs = weight_ptr + (o + offs_o)[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
        
        mask_dout = (offs_b[:, None] < B) & ((o + offs_o)[None, :] < O)
        mask_weight = ((o + offs_o)[:, None] < O) & (offs_i[None, :] < I)
        
        dout = tl.load(dout_ptrs, mask=mask_dout, other=0.0)
        weight = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(dout, weight, allow_tf32=True)
    
    din_ptrs = d_input_ptr + offs_b[:, None] * stride_din_b + offs_i[None, :] * stride_din_i
    mask = (offs_b[:, None] < B) & (offs_i[None, :] < I)
    tl.store(din_ptrs, acc, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def backward_weight_kernel(
    input_ptr, d_output_ptr, d_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_dout_b, stride_dout_o,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr
):
    """
    Computes gradient of weight matrix for linear layer.

    Args:
        input_ptr (tensor): Pointer to input tensor (B, I)
        d_output_ptr (tensor): Pointer to output gradient tensor (B, O)
        d_weight_ptr (tensor): Pointer to weight gradient tensor (O, I)
        B, I, O (int): Dimension sizes
        stride_*: Memory stride information
        BLOCK_SIZE_O, BLOCK_SIZE_I (tl.constexpr): Block sizes for autotuning

    Explanation:
        1. Initialize block indices for output and input dimensions
        2. Create memory pointers with appropriate offsets
        3. Iterate over batch dimension with atomic accumulation
        4. Store final accumulated gradients
    """
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = tl.arange(0, 128)
    
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    for b in range(0, B, 128):
        input_ptrs = input_ptr + (b + offs_b)[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
        dout_ptrs = d_output_ptr + (b + offs_b)[:, None] * stride_dout_b + offs_o[None, :] * stride_dout_o
        
        mask_input = ((b + offs_b)[:, None] < B) & (offs_i[None, :] < I)
        mask_dout = ((b + offs_b)[:, None] < B) & (offs_o[None, :] < O)
        
        input_block = tl.load(input_ptrs, mask=mask_input, other=0.0)
        dout_block = tl.load(dout_ptrs, mask=mask_dout, other=0.0)
        
        acc += tl.dot(dout_block.T, input_block, allow_tf32=True)
    
    d_weight_ptrs = d_weight_ptr + offs_o[:, None] * O + offs_i[None, :]
    mask = (offs_o[:, None] < O) & (offs_i[None, :] < I)
    tl.atomic_add(d_weight_ptrs, acc, mask=mask)

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O = weight.shape[0]
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
        )
        
        forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1)
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O = weight.shape[0]
        grad_input = torch.empty_like(input)
        grad_weight = torch.zeros_like(weight)
        
        # Compute grad_input
        grid_input = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        backward_input_kernel[grid_input](
            grad_output, weight, grad_input,
            B, I, O,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1)
        )
        
        # Compute grad_weight
        grid_weight = lambda meta: (
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        backward_weight_kernel[grid_weight](
            input, grad_output, grad_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            grad_output.stride(0), grad_output.stride(1)
        )
        
        return grad_input, grad_weight

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.kaiming_normal_(self.weight)
        
    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

if __name__ == "__main__":
    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    def module_generator(B, I, O, device='cuda', dtype=torch.float16):
        triton_module = TritonLinear(I, O, device=device, dtype=dtype)
        torch_module = torch.nn.Linear(I, O, bias=False, device=device, dtype=dtype)
        torch_module.weight.data = triton_module.weight.data.clone()
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
To test the TritonLinear implementation, run the following command (assuming CUDA-enabled device and Triton installed):

```bash
python triton_linear.py
```
</test>