Timestamp: 20250406_055502

<think>
Okay, I need to create a Triton module for PyTorch's nn.Linear, implementing both forward and backward passes. The filename should be linear_triton.py. The user mentioned using autotune for block sizes (BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O) in both kernels and using super blocks with corresponding GROUP_SIZE_*. Also, I need to include the provided test code in the main block.

First, I'll start by understanding what TritonLinear should do. The PyTorch Linear layer applies a linear transformation: output = input @ weight.T + bias. Since the user specified bias=False in the test code, maybe I can ignore bias for now.

For the forward pass, the Triton kernel needs to handle matrix multiplication. The input has shape (B, I), and the weight is (O, I), so the output is (B, O). The kernel should efficiently compute this product using Triton's tiling and autotuning.

In Triton, autotuning involves defining configurable parameters like BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O. These block sizes determine how the matrices are split into tiles for computation. The GROUP_SIZE parameters are used for super-blocking, which groups multiple blocks together to improve memory access patterns.

The forward kernel will have nested loops over the batch dimension (B) and the input dimension (I). Each block processes a tile of the input and weight matrices, accumulating the result into the output. Autotuning will test different block sizes to find the fastest configuration.

For the backward pass, there are two gradients to compute: d_input (gradient of input) and d_weight (gradient of weight). The backward kernel for d_input involves multiplying d_output with the weight matrix: d_input = d_output @ weight. The kernel for d_weight is a bit more complex, as it's the outer product of input and d_output: d_weight = d_output.T @ input. Again, these operations need to be optimized with tiling and autotuning.

The autotune decorator in Triton allows specifying multiple values for each block size. The tuner will benchmark each combination to select the best-performing one. Super-blocking (GROUP_SIZE) helps in coalescing memory accesses by processing multiple blocks together, which can improve memory bandwidth utilization.

I need to structure the TritonLinear class to hold the weight parameter, similar to PyTorch's Linear. The forward method will call the triton_linear_forward function, and the backward method (or autograd Function) will call the triton_linear_backward function.

Wait, PyTorch modules require defining the backward pass via autograd Functions. So, I should create a custom autograd Function that encapsulates both forward and backward Triton kernels. The TritonLinear module will use this Function in its forward pass.

So, the TritonLinear class will have a weight tensor. In the forward method, it will take input, and use the custom Function to compute output, saving necessary tensors for backward. The Function's backward method will compute the gradients using the Triton kernels.

Now, implementing the kernels. For the forward kernel:

- Input is (B, I), weight is (O, I), output is (B, O).
- Each block processes a tile of B and O dimensions.
- Load input and weight tiles, compute the dot product, accumulate into output.

For the backward kernels:

- d_input computation: d_output (B, O) @ weight (O, I) → (B, I)
- d_weight computation: d_output.T (O, B) @ input (B, I) → (O, I)

Each backward kernel will have its own Triton function with autotuned block sizes.

I need to define the Triton kernels with proper indexing and memory operations. Using tl.dot for matrix multiplications within tiles, and accumulating results.

Testing the code requires comparing against PyTorch's Linear layer, both in correctness (output and gradients matching) and performance. The provided main block uses functions from benchmark.py to do this, so the TritonLinear must have the same interface as torch.nn.Linear.

Potential issues to consider: memory layout (row-major vs column-major), handling different data types (the test uses float16), and ensuring that autotuning correctly selects optimal parameters across different input sizes.

Now, putting this all together into the linear_triton.py file. The code should include the TritonLinear class, the custom autograd Function, and the Triton kernels with autotuning. The main block will execute the correctness and performance tests as specified.
</think>

```python
# filename: linear_triton.py

import torch
import triton
import triton.language as tl
from torch.autograd import Function

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 512, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 512, 'BLOCK_SIZE_O': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 512}, num_warps=4),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def triton_linear_forward(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_O: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr
):
    """
    Compute forward pass for linear layer using Triton.

    Args:
        input_ptr (tensor): Pointer to input tensor of shape (B, I)
        weight_ptr (tensor): Pointer to weight tensor of shape (O, I)
        output_ptr (tensor): Pointer to output tensor of shape (B, O)
        B (int): Batch dimension size
        I (int): Input feature dimension size
        O (int): Output feature dimension size
        GROUP_SIZE_* (tl.constexpr): Super block grouping sizes
        BLOCK_SIZE_* (tl.constexpr): Block sizes for tiling

    Explanation:
        1. Create program IDs for super blocks
        2. Load input and weight tiles from global memory
        3. Compute partial dot products using tensor cores
        4. Accumulate results across tiles
        5. Store final results to output tensor
    """
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    pid_b = pid // num_pid_o
    pid_o = pid % num_pid_o

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)

    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i

    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, BLOCK_SIZE_I * GROUP_SIZE_I):
        input = tl.load(input_ptrs, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I - i), other=0.0)
        weight = tl.load(weight_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I - i), other=0.0)
        acc += tl.dot(input, weight, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_I * GROUP_SIZE_I * stride_input_i
        weight_ptrs += BLOCK_SIZE_I * GROUP_SIZE_I * stride_weight_i

    acc = acc.to(tl.float16)
    offs_output = offs_b[:, None] * stride_output_b + offs_o[None, :] * stride_output_o
    output_ptrs = output_ptr + offs_output
    tl.store(output_ptrs, acc, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 512, 'BLOCK_SIZE_O': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 512}, num_warps=4),
    ],
    key=['B', 'O']
)
@triton.jit
def triton_linear_backward_input(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, I, O,
    stride_dout_b, stride_dout_o,
    stride_weight_o, stride_weight_i,
    stride_din_b, stride_din_i,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_O: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr
):
    """
    Compute gradient for input in linear layer backward pass.

    Args:
        d_output_ptr (tensor): Pointer to output gradient tensor of shape (B, O)
        weight_ptr (tensor): Pointer to weight tensor of shape (O, I)
        d_input_ptr (tensor): Pointer to input gradient tensor of shape (B, I)
        B (int): Batch dimension size
        I (int): Input feature dimension size
        O (int): Output feature dimension size
        GROUP_SIZE_* (tl.constexpr): Super block grouping sizes
        BLOCK_SIZE_* (tl.constexpr): Block sizes for tiling

    Explanation:
        1. Create program IDs for super blocks
        2. Load d_output and weight tiles
        3. Compute gradient contributions using matrix multiplication
        4. Accumulate and store results to d_input
    """
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_O)
    pid_b = pid // num_pid_i
    pid_i = pid % num_pid_i

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_o = tl.arange(0, BLOCK_SIZE_O)

    dout_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + offs_o[None, :] * stride_dout_o
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i

    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for o in range(0, O, BLOCK_SIZE_O * GROUP_SIZE_O):
        dout = tl.load(dout_ptrs, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O - o), other=0.0)
        weight = tl.load(weight_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I - o), other=0.0)
        acc += tl.dot(dout, weight, allow_tf32=True)
        dout_ptrs += BLOCK_SIZE_O * GROUP_SIZE_O * stride_dout_o
        weight_ptrs += BLOCK_SIZE_O * GROUP_SIZE_O * stride_weight_o

    acc = acc.to(tl.float16)
    offs_din = offs_b[:, None] * stride_din_b + offs_i[None, :] * stride_din_i
    din_ptrs = d_input_ptr + offs_din
    tl.store(din_ptrs, acc, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 512, 'BLOCK_SIZE_O': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 512}, num_warps=4),
    ],
    key=['I', 'O']
)
@triton.jit
def triton_linear_backward_weight(
    input_ptr, d_output_ptr, d_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_dout_b, stride_dout_o,
    stride_dw_o, stride_dw_i,
    GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_O: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr
):
    """
    Compute gradient for weight in linear layer backward pass.

    Args:
        input_ptr (tensor): Pointer to input tensor of shape (B, I)
        d_output_ptr (tensor): Pointer to output gradient tensor of shape (B, O)
        d_weight_ptr (tensor): Pointer to weight gradient tensor of shape (O, I)
        B (int): Batch dimension size
        I (int): Input feature dimension size
        O (int): Output feature dimension size
        GROUP_SIZE_* (tl.constexpr): Super block grouping sizes
        BLOCK_SIZE_* (tl.constexpr): Block sizes for tiling

    Explanation:
        1. Create program IDs for super blocks
        2. Load input and d_output tiles
        3. Compute outer product contributions for weight gradient
        4. Accumulate and store results to d_weight
    """
    pid = tl.program_id(0)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    pid_i = pid // num_pid_o
    pid_o = pid % num_pid_o

    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_b = tl.arange(0, BLOCK_SIZE_I)

    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    dout_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + offs_o[None, :] * stride_dout_o

    acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_O), dtype=tl.float32)
    
    for b in range(0, B, BLOCK_SIZE_I * GROUP_SIZE_I):
        inp = tl.load(input_ptrs, mask=(offs_b[:, None] < B - b) & (offs_i[None, :] < I), other=0.0)
        dout = tl.load(dout_ptrs, mask=(offs_b[:, None] < B - b) & (offs_o[None, :] < O), other=0.0)
        acc += tl.dot(inp, dout, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_I * GROUP_SIZE_I * stride_input_b
        dout_ptrs += BLOCK_SIZE_I * GROUP_SIZE_I * stride_dout_b

    acc = acc.to(tl.float16)
    offs_dw = offs_o[:, None] * stride_dw_o + offs_i[None, :] * stride_dw_i
    dw_ptrs = d_weight_ptr + offs_dw
    tl.atomic_add(dw_ptrs, acc, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I))

class TritonLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight):
        B, I = input.shape
        O, _ = weight.shape
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        triton_linear_forward[(B * O,)](input, weight, output,
                                      B, I, O,
                                      input.stride(0), input.stride(1),
                                      weight.stride(0), weight.stride(1),
                                      output.stride(0), output.stride(1),
                                      GROUP_SIZE_B=2, GROUP_SIZE_I=2, GROUP_SIZE_O=2)
        
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O, _ = weight.shape
        
        d_input = torch.empty_like(input)
        d_weight = torch.zeros_like(weight)

        triton_linear_backward_input[(B * I,)](d_output, weight, d_input,
                                             B, I, O,
                                             d_output.stride(0), d_output.stride(1),
                                             weight.stride(0), weight.stride(1),
                                             d_input.stride(0), d_input.stride(1),
                                             GROUP_SIZE_B=2, GROUP_SIZE_O=2)
        
        triton_linear_backward_weight[(I * O,)](input, d_output, d_weight,
                                              B, I, O,
                                              input.stride(0), input.stride(1),
                                              d_output.stride(0), d_output.stride(1),
                                              d_weight.stride(0), d_weight.stride(1),
                                              GROUP_SIZE_I=2, GROUP_SIZE_O=2)
        
        return d_input, d_weight

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity='linear')

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
To test the TritonLinear implementation, run the following command. Ensure you have PyTorch and Triton installed with CUDA support.

```bash
python linear_triton.py
```
This will execute the correctness checks and performance benchmarks as defined in the main block. The results will be saved as JSON files and performance plots.
</test>