Timestamp: 20250406_055345

<think>
Okay, I need to create a Triton module for PyTorch's nn.Linear, implementing both forward and backward passes with autotuning for block sizes. Let me start by understanding the requirements.

First, the TritonLinear class should mirror the functionality of torch.nn.Linear. That means it needs to handle input tensors, apply a weight matrix, and compute the output. Since there's no bias mentioned, I'll omit that part.

The forward and backward kernels need to use Triton's autotuning. Autotuning in Triton involves specifying configurable parameters like BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O, and GROUP_SIZE_*. These block sizes determine how the computation is partitioned for efficiency. Using super blocks might help in managing memory access patterns better, so I should structure the kernels to take advantage of that.

For the forward pass, the matrix multiplication is between the input (B x I) and the weight (O x I), resulting in an output (B x O). The Triton kernel will need to tile these matrices into blocks. Autotuning will select the best block sizes based on the problem dimensions. I should use triton.Config to define different combinations of block sizes and group sizes, then let the autotuner pick the fastest one.

The backward pass involves computing gradients for the input (dx) and the weights (dw). The dx is computed by multiplying d_output (B x O) with the weight matrix (O x I), resulting in (B x I). The dw is computed by multiplying the transposed input (I x B) with d_output (B x O), resulting in (I x O). Each of these operations will need their own Triton kernels with autotuned parameters.

In the TritonLinear class, during forward, I'll save the input for use in the backward pass. The backward function will compute both dx and dw using the saved input and the incoming gradient d_output.

Now, structuring the code:

1. Import necessary modules: torch, triton, triton.language as tl.
2. Define the TritonLinear class, inheriting from torch.nn.Module.
3. Initialize the weight parameter with the correct dimensions (O x I) and data type.
4. Implement the forward method, which calls the triton_linear_forward function.
5. Implement the backward method in a custom Function class, which calls both triton_linear_backward_input and triton_linear_backward_weight functions.
6. Define the Triton kernels with @triton.autotune for autotuning configurations and @triton.jit for JIT compilation.

For each kernel, I need to write detailed comments as per the template: Description, Args, Returns, Explanation. Let's ensure each argument's type and shape are specified, and the steps are clearly listed.

Testing the code using the provided main block. The module_generator creates TritonLinear and torch.nn.Linear instances. The forward and backward generators create inputs and gradients. The benchmark functions from benchmark.py will verify correctness and measure performance.

Potential issues to consider: memory layout (row-major vs column-major), correct handling of tensor dimensions in Triton kernels, and ensuring that autotune configurations cover a sufficient range of block sizes. Also, need to make sure that during backward, the saved input is correctly used.

I should check the matrix multiplication dimensions in both forward and backward passes. For example, in the forward pass, input (B, I) multiplied by weight (O, I).T becomes (B, O). In the backward for dw, the input is transposed to (I, B) and multiplied by d_output (B, O) to get (I, O), which matches the weight gradient shape.

Autotune configurations should include different block sizes for B, I, O dimensions. Using powers of two is common for GPU efficiency. The GROUP_SIZE parameters might control how groups of threads cooperate, which can affect memory access patterns and occupancy.

Testing the code will involve running the provided main block, which calls verify_correctness and benchmark_performance functions. Any discrepancies in outputs between Triton and PyTorch implementations should be checked with tolerances.

Finally, ensure that the code is properly formatted with the required comments and that all functions are correctly annotated as per the template.
</think>

```python
# filename: linear_triton.py

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 64}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def triton_linear_forward(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_ib, stride_ii,
    stride_wo, stride_wi,
    stride_ob, stride_oo,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr = 8
):
    """
    Computes forward pass of linear layer using block matrix multiplication.

    Args:
        input_ptr (tl.pointer): Input tensor pointer [B, I]
        weight_ptr (tl.pointer): Weight tensor pointer [O, I]
        output_ptr (tl.pointer): Output tensor pointer [B, O]
        B (int): Batch dimension size
        I (int): Input feature dimension size
        O (int): Output feature dimension size
        stride_ib (int): Input batch stride
        stride_ii (int): Input feature stride
        stride_wo (int): Weight output stride
        stride_wi (int): Weight input stride
        stride_ob (int): Output batch stride
        stride_oo (int): Output feature stride
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension super blocks

    Returns:
        None: Stores result in output_ptr

    Explanation:
        1. Calculate block indices for batch and output dimensions
        2. Load blocks of input and weight matrices into shared memory
        3. Compute partial dot products using matrix multiplication
        4. Accumulate results across blocks and store in output
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, BLOCK_SIZE_I * GROUP_SIZE_B):
        offs_i = i + tl.arange(0, BLOCK_SIZE_I)
        
        mask_input = (offs_b[:, None] < B) & (offs_i[None, :] < I)
        input_block = tl.load(input_ptr + offs_b[:, None] * stride_ib + offs_i[None, :] * stride_ii,
                             mask=mask_input, other=0.0)
        
        mask_weight = (offs_o[:, None] < O) & (offs_i[None, :] < I)
        weight_block = tl.load(weight_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi,
                              mask=mask_weight, other=0.0)
        
        acc += tl.dot(input_block, weight_block, allow_tf32=True)
    
    mask_output = (offs_b[:, None] < B) & (offs_o[None, :] < O)
    tl.store(output_ptr + offs_b[:, None] * stride_ob + offs_o[None, :] * stride_oo,
            acc.to(output_ptr.dtype.element_ty), mask=mask_output)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 64}, num_warps=8),
    ],
    key=['B', 'O', 'I']
)
@triton.jit
def triton_linear_backward_input(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, O, I,
    stride_dob, stride_doo,
    stride_wo, stride_wi,
    stride_dib, stride_dii,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr = 8
):
    """
    Computes input gradient for linear layer backward pass.

    Args:
        d_output_ptr (tl.pointer): Output gradient pointer [B, O]
        weight_ptr (tl.pointer): Weight tensor pointer [O, I]
        d_input_ptr (tl.pointer): Input gradient pointer [B, I]
        B (int): Batch dimension size
        O (int): Output feature dimension size
        I (int): Input feature dimension size
        stride_dob (int): Output gradient batch stride
        stride_doo (int): Output gradient feature stride
        stride_wo (int): Weight output stride
        stride_wi (int): Weight input stride
        stride_dib (int): Input gradient batch stride
        stride_dii (int): Input gradient feature stride
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_O (tl.constexpr): Group size for output dimension super blocks

    Returns:
        None: Stores result in d_input_ptr

    Explanation:
        1. Calculate block indices for batch and input dimensions
        2. Load blocks of d_output and weight matrices
        3. Compute gradient contributions through matrix multiplication
        4. Accumulate and store results in input gradient tensor
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * GROUP_SIZE_O * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for o in range(0, O, BLOCK_SIZE_O * GROUP_SIZE_O):
        offs_o = o + tl.arange(0, BLOCK_SIZE_O)
        
        mask_dout = (offs_b[:, None] < B) & (offs_o[None, :] < O)
        dout_block = tl.load(d_output_ptr + offs_b[:, None] * stride_dob + offs_o[None, :] * stride_doo,
                            mask=mask_dout, other=0.0)
        
        mask_weight = (offs_o[:, None] < O) & (offs_i[None, :] < I)
        weight_block = tl.load(weight_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi,
                              mask=mask_weight, other=0.0)
        
        acc += tl.dot(dout_block, weight_block, allow_tf32=True)
    
    mask_din = (offs_b[:, None] < B) & (offs_i[None, :] < I)
    tl.store(d_input_ptr + offs_b[:, None] * stride_dib + offs_i[None, :] * stride_dii,
            acc.to(d_input_ptr.dtype.element_ty), mask=mask_din)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 64}, num_warps=8),
    ],
    key=['I', 'O', 'B']
)
@triton.jit
def triton_linear_backward_weight(
    input_ptr, d_output_ptr, d_weight_ptr,
    I, O, B,
    stride_ib, stride_ii,
    stride_dob, stride_doo,
    stride_dwo, stride_dwi,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr = 8
):
    """
    Computes weight gradient for linear layer backward pass.

    Args:
        input_ptr (tl.pointer): Input tensor pointer [B, I]
        d_output_ptr (tl.pointer): Output gradient pointer [B, O]
        d_weight_ptr (tl.pointer): Weight gradient pointer [O, I]
        I (int): Input feature dimension size
        O (int): Output feature dimension size
        B (int): Batch dimension size
        stride_ib (int): Input batch stride
        stride_ii (int): Input feature stride
        stride_dob (int): Output gradient batch stride
        stride_doo (int): Output gradient feature stride
        stride_dwo (int): Weight gradient output stride
        stride_dwi (int): Weight gradient input stride
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension super blocks

    Returns:
        None: Stores result in d_weight_ptr

    Explanation:
        1. Calculate block indices for input and output dimensions
        2. Load blocks of input and d_output matrices
        3. Compute outer product contributions for weight gradients
        4. Atomic accumulate results into weight gradient tensor
    """
    pid_i = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    for b in range(0, B, BLOCK_SIZE_I * GROUP_SIZE_B):
        offs_b = b + tl.arange(0, BLOCK_SIZE_I)
        
        mask_input = (offs_b[:, None] < B) & (offs_i[None, :] < I)
        input_block = tl.load(input_ptr + offs_b[:, None] * stride_ib + offs_i[None, :] * stride_ii,
                             mask=mask_input, other=0.0)
        
        mask_dout = (offs_b[:, None] < B) & (offs_o[None, :] < O)
        dout_block = tl.load(d_output_ptr + offs_b[:, None] * stride_dob + offs_o[None, :] * stride_doo,
                            mask=mask_dout, other=0.0)
        
        acc += tl.dot(dout_block, input_block, allow_tf32=True)
    
    mask_dw = (offs_o[:, None] < O) & (offs_i[None, :] < I)
    tl.atomic_add(d_weight_ptr + offs_o[:, None] * stride_dwo + offs_i[None, :] * stride_dwi,
                 acc.to(d_weight_ptr.dtype.element_ty), mask=mask_dw)


class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O, _ = weight.shape
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        grid = lambda opt: (
            triton.cdiv(B, opt['BLOCK_SIZE_B']),
            triton.cdiv(O, opt['BLOCK_SIZE_O']),
        )
        
        triton_linear_forward[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1)
        )
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O, _ = weight.shape
        
        d_input = torch.empty_like(input)
        d_weight = torch.zeros_like(weight)
        
        # Backward input
        grid_input = lambda opt: (
            triton.cdiv(B, opt['BLOCK_SIZE_B']),
            triton.cdiv(I, opt['BLOCK_SIZE_O'] * opt['GROUP_SIZE_O']),
        )
        triton_linear_backward_input[grid_input](
            d_output, weight, d_input,
            B, O, I,
            d_output.stride(0), d_output.stride(1),
            weight.stride(0), weight.stride(1),
            d_input.stride(0), d_input.stride(1)
        )
        
        # Backward weight
        grid_weight = lambda opt: (
            triton.cdiv(I, opt['BLOCK_SIZE_I']),
            triton.cdiv(O, opt['BLOCK_SIZE_O']),
        )
        triton_linear_backward_weight[grid_weight](
            input, d_output, d_weight,
            I, O, B,
            input.stride(0), input.stride(1),
            d_output.stride(0), d_output.stride(1),
            d_weight.stride(0), d_weight.stride(1)
        )
        
        return d_input, d_weight


class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)


if __name__ == "__main__":
    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    def module_generator(B, I, O, device='cuda', dtype=torch.float16):
        triton_module = TritonLinear(I, O, device='cuda', dtype=torch.float16)
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

This will execute the correctness checks and performance benchmarks for both forward and backward passes across different tensor sizes. The results will be saved as JSON files and performance plots.
</test>