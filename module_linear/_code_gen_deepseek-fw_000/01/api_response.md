Timestamp: 20250315_191505

<think>
Okay, I need to create a Triton-based implementation of PyTorch's nn.Linear module, including both forward and backward passes. The user specified using autotune for block sizes and super blocks, so I should remember to set up the Triton kernels with those parameters. Let me start by understanding the problem.

First, the TritonLinear class needs to mimic the functionality of torch.nn.Linear. That means during forward, it takes an input tensor and applies a linear transformation using weights. The backward pass will compute gradients for both the input and the weights.

For the forward kernel, matrix multiplication is the core operation. I'll need to write a Triton kernel that handles this efficiently. Autotuning block sizes (BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O) is required. Using Triton's autotune configs, I can define different combinations of these block sizes and let Triton select the best one based on performance.

In the forward kernel, each block should handle a portion of the batch (B), input features (I), and output features (O). The input matrix is of shape (B, I), and the weight matrix is (O, I), so the output is (B, O). The kernel will iterate over these dimensions in blocks, using matrix operations within each block.

For the backward pass, there are two gradients to compute: d_input (gradient of the loss with respect to the input) and d_weight (gradient with respect to the weights). The d_input is computed by multiplying d_output with the weight matrix, and d_weight is computed by multiplying the input transpose with d_output. Each of these operations will need their own Triton kernels with autotuned block sizes as well.

In the backward kernel for d_input, the operation is similar to the forward pass but using the weight matrix and d_output. The d_weight kernel will need to handle the transpose of the input and multiply it by d_output. Here, using super blocks might help in optimizing memory access patterns, so I'll need to consider how to tile the matrices effectively.

I need to structure the Triton kernels with the correct grid dimensions and block sizes. For autotuning, I'll use the @triton.autotune decorator, providing lists of possible values for each block size and group size. The configs will define different combinations, and Triton will benchmark them to find the fastest.

Another consideration is handling the data types correctly. The user's benchmark code uses float16, so the kernels should be compatible with that. Also, the input and weight tensors should be properly initialized, and in the TritonLinear class, the weights need to be registered as a parameter so that PyTorch's autograd can track them.

For the forward method in TritonLinear, the input is processed, and the forward kernel is called. Similarly, during backward, the saved tensors from the forward pass (input and weight) are used to compute the gradients. I'll need to implement the backward function using torch.autograd.Function, which holds the forward and backward static methods.

Testing is done via the provided benchmark code, which checks correctness and performance. The verify functions compare the Triton outputs with PyTorch's native implementation, so the Triton kernels must produce numerically close results, considering floating-point tolerances.

Potential issues I might encounter include incorrect block sizing leading to out-of-bounds memory access, which can be mitigated by checking the ranges in the kernel. Also, ensuring that the autotune configurations cover a sufficient range of block sizes to find the optimal performance on different hardware.

I need to ensure that the Triton kernels are compatible with the dimensions specified in B_sizes, I_sizes, and O_sizes. The grid dimensions in each kernel should be calculated based on the block sizes and the problem dimensions to cover the entire matrix.

Finally, integrating everything into the TritonLinear class with proper parameter handling and ensuring that the backward pass correctly computes both gradients. The autotuning parameters should be applied to both forward and backward kernels, and the super block grouping for memory efficiency.

Let me outline the steps for each kernel:

Forward Kernel:
1. Load input and weight blocks into shared memory.
2. Compute the dot product for the relevant elements in the block.
3. Accumulate results and store in the output tensor.

Backward d_input Kernel:
1. Use d_output and weights to compute the gradient of the input.
2. Similar structure to the forward kernel but with d_output as input.

Backward d_weight Kernel:
1. Multiply the transpose of the input with d_output to compute weight gradients.
2. Need to handle the transpose efficiently, possibly using super blocks for better memory access.

By structuring the kernels with these steps and applying autotuning, the TritonLinear module should achieve performance comparable or better than PyTorch's native implementation, which is then verified by the benchmark scripts.
</think>

```python
# filename: linear_triton.py

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr
):
    """
    Computes forward pass for linear layer using block matrix multiplication.

    Args:
        input_ptr (tensor): Input tensor of shape (B, I)
        weight_ptr (tensor): Weight tensor of shape (O, I)
        output_ptr (tensor): Output tensor of shape (B, O)
        B (int): Batch size dimension
        I (int): Input feature dimension
        O (int): Output feature dimension
        stride_*: Memory strides for corresponding dimensions
        BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O (tl.constexpr): Block sizes for each dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension super-blocking

    Explanation:
        1. Calculate block indices for batch and output dimensions
        2. Load blocks of input and weight matrices into SRAM
        3. Compute matrix product for current blocks
        4. Accumulate results and write back to output
    """
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    pid_b = pid // num_pid_o
    pid_o = pid % num_pid_o

    offs_b = (pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)) % B
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)

    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, BLOCK_SIZE_I * GROUP_SIZE_B):
        input_block = tl.load(input_ptrs, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I - i), other=0.0)
        weight_block = tl.load(weight_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I - i), other=0.0)
        accumulator += tl.dot(input_block, weight_block, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_I * GROUP_SIZE_B * stride_input_i
        weight_ptrs += BLOCK_SIZE_I * GROUP_SIZE_B * stride_weight_i

    offs_output_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_output_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    output_ptrs = output_ptr + offs_output_b[:, None] * stride_output_b + offs_output_o[None, :] * stride_output_o
    output_mask = (offs_output_b[:, None] < B) & (offs_output_o[None, :] < O)
    tl.store(output_ptrs, accumulator.to(tl.float16), mask=output_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}, num_warps=8),
    ],
    key=['B', 'O', 'I']
)
@triton.jit
def linear_backward_dinput_kernel(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, O, I,
    stride_dout_b, stride_dout_o,
    stride_weight_o, stride_weight_i,
    stride_din_b, stride_din_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr
):
    """
    Computes gradient of input for linear layer backward pass.

    Args:
        d_output_ptr (tensor): Gradient of output (B, O)
        weight_ptr (tensor): Weight tensor (O, I)
        d_input_ptr (tensor): Gradient of input (B, I)
        B, O, I (int): Dimensions of tensors
        stride_*: Memory strides for corresponding dimensions
        BLOCK_SIZE_B, BLOCK_SIZE_O (tl.constexpr): Block sizes
        GROUP_SIZE_B (tl.constexpr): Group size for super-blocking

    Explanation:
        1. Calculate block indices for batch and input dimensions
        2. Load blocks of d_output and weight matrices
        3. Compute matrix product for gradient calculation
        4. Accumulate and store results in d_input
    """
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_O)  # Reusing BLOCK_SIZE_O for input dimension
    pid_b = pid // num_pid_i
    pid_i = pid % num_pid_i

    offs_b = (pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)) % B
    offs_i = pid_i * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_o = tl.arange(0, BLOCK_SIZE_O)

    dout_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + offs_o[None, :] * stride_dout_o
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for o in range(0, O, BLOCK_SIZE_O * GROUP_SIZE_B):
        dout_block = tl.load(dout_ptrs, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O - o), other=0.0)
        weight_block = tl.load(weight_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I), other=0.0)
        accumulator += tl.dot(dout_block, weight_block, allow_tf32=True)
        dout_ptrs += BLOCK_SIZE_O * GROUP_SIZE_B * stride_dout_o
        weight_ptrs += BLOCK_SIZE_O * GROUP_SIZE_B * stride_weight_o

    offs_din_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_din_i = pid_i * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    din_ptrs = d_input_ptr + offs_din_b[:, None] * stride_din_b + offs_din_i[None, :] * stride_din_i
    din_mask = (offs_din_b[:, None] < B) & (offs_din_i[None, :] < I)
    tl.store(din_ptrs, accumulator.to(tl.float16), mask=din_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_dweight_kernel(
    input_ptr, d_output_ptr, d_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_dout_b, stride_dout_o,
    stride_dw_o, stride_dw_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr
):
    """
    Computes gradient of weights for linear layer backward pass.

    Args:
        input_ptr (tensor): Input tensor (B, I)
        d_output_ptr (tensor): Gradient of output (B, O)
        d_weight_ptr (tensor): Gradient of weights (O, I)
        B, I, O (int): Dimensions of tensors
        stride_*: Memory strides for corresponding dimensions
        BLOCK_SIZE_B, BLOCK_SIZE_I (tl.constexpr): Block sizes
        GROUP_SIZE_B (tl.constexpr): Group size for super-blocking

    Explanation:
        1. Calculate block indices for output and input dimensions
        2. Load blocks of input and d_output matrices
        3. Compute outer product for gradient accumulation
        4. Atomically update d_weight tensor
    """
    pid = tl.program_id(0)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_B)  # Reuse BLOCK_SIZE_B for O
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
    pid_o = pid // num_pid_i
    pid_i = pid % num_pid_i

    offs_o = pid_o * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = tl.arange(0, BLOCK_SIZE_B)

    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    dout_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + offs_o[None, :] * stride_dout_o

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for b in range(0, B, BLOCK_SIZE_B * GROUP_SIZE_B):
        input_block = tl.load(input_ptrs, mask=(offs_b[:, None] < B - b) & (offs_i[None, :] < I), other=0.0)
        dout_block = tl.load(dout_ptrs, mask=(offs_b[:, None] < B - b) & (offs_o[None, :] < O), other=0.0)
        accumulator += tl.dot(dout_block, input_block, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_B * GROUP_SIZE_B * stride_input_b
        dout_ptrs += BLOCK_SIZE_B * GROUP_SIZE_B * stride_dout_b

    dw_ptrs = d_weight_ptr + offs_o[:, None] * stride_dw_o + offs_i[None, :] * stride_dw_i
    dw_mask = (offs_o[:, None] < O) & (offs_i[None, :] < I)
    tl.atomic_add(dw_ptrs, accumulator.to(tl.float16), mask=dw_mask)

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O = weight.shape[0]
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(O, meta['BLOCK_SIZE_O']),)
        linear_forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O = weight.shape[0]

        d_input = torch.empty_like(input)
        d_weight = torch.zeros_like(weight)

        # Compute d_input
        grid_dinput = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_O']),)
        linear_backward_dinput_kernel[grid_dinput](
            d_output, weight, d_input,
            B, O, I,
            d_output.stride(0), d_output.stride(1),
            weight.stride(0), weight.stride(1),
            d_input.stride(0), d_input.stride(1),
        )

        # Compute d_weight
        grid_dweight = lambda meta: (triton.cdiv(O, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_I']),)
        linear_backward_dweight_kernel[grid_dweight](
            input, d_output, d_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            d_output.stride(0), d_output.stride(1),
            d_weight.stride(0), d_weight.stride(1),
        )

        return d_input, d_weight

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

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
To verify the implementation and benchmark performance, execute the following commands:

```bash
python linear_triton.py
```

This will:
1. Run correctness checks comparing outputs with PyTorch's implementation
2. Generate performance benchmarks for forward and backward passes
3. Save results to JSON files (code.correctness*.json, code.performance*.json)
</test>