Timestamp: 20250406_055530

<think>
Okay, I need to create a Triton implementation of the PyTorch nn.Linear module, including both forward and backward passes. The user specified using autotune for block sizes (BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O) in both kernels and using super blocks with corresponding GROUP_SIZE parameters. Also, I have to structure the code according to the given template with detailed comments.

First, let's think about the forward pass. The linear layer's forward operation is a matrix multiplication between the input and the weight matrix, possibly adding a bias. Since we're implementing this in Triton, I'll need to write a Triton kernel that performs this operation efficiently. Autotuning the block sizes is crucial for optimizing performance across different input sizes. The kernel should handle different block configurations, and Triton's autotune feature can help select the best ones.

For the backward pass, I need to compute the gradients of the input (dx) and the weights (dw). The backward pass involves two main operations: computing dx by multiplying the gradient of the output (d_output) with the weight matrix, and computing dw by multiplying the transposed input with d_output. Each of these operations will require their own Triton kernels, also with autotuned block sizes.

The user mentioned using super blocks and autotuning GROUP_SIZE_*. Super blocks are a way to group multiple blocks together to improve memory access patterns and parallelism. I'll need to define these in the autotune configurations, possibly varying the group sizes for different dimensions (B, I, O).

Looking at the provided benchmark code, the TritonLinear module should have the same interface as torch.nn.Linear. So, the class should initialize weights, handle bias optionally, and define the forward method. The backward method will be handled by defining the backward pass in the Triton kernels.

In the forward kernel, the input is of shape (B, I), and the weight is (O, I). The output is (B, O). The kernel will load blocks of input and weight, compute the dot product, and store the result. Autotuning parameters will include BLOCK_SIZE_B (batch dimension), BLOCK_SIZE_I (input dimension), and BLOCK_SIZE_O (output dimension). Group sizes might control how many blocks are processed together to optimize memory access.

For the backward kernels, the dx kernel computes the gradient of the input by multiplying d_output (B, O) with weight (O, I), resulting in dx (B, I). The dw kernel computes the gradient of the weight by multiplying the transposed input (I, B) with d_output (B, O), resulting in dw (I, O). Each of these will have their own block sizes and group sizes to autotune.

I need to ensure that the Triton kernels are correctly handling the data types and memory layouts. Using pointers with the correct shapes and strides is essential. Also, since the benchmark uses fp16, the kernels should be optimized for half-precision operations.

Testing is handled by the benchmark.py code, which compares the Triton implementation against PyTorch's implementation for correctness and performance. The TritonLinear module must be compatible with the module_generator function, which creates both the Triton and PyTorch modules for comparison.

Potential issues include incorrect kernel configurations leading to mismatched results, especially with different block sizes. Autotuning might require a range of possible configurations to find the optimal one. Also, handling the bias term is mentioned, but the current benchmark code sets bias=False, so maybe the initial implementation can omit bias and focus on the weight matrix.

I should structure the TritonLinear class to initialize the weight parameter, similar to torch.nn.Linear. The forward method will call the Triton kernel, and the backward method will be handled by the defined backward kernels. Using Triton's autotune decorator with configs that specify different block and group sizes will allow the kernels to be optimized during the tuning phase.

Now, putting this all together into code. Each kernel (forward, dx, dw) will have its own function with the appropriate Triton annotations and autotuning configurations. The TritonLinear class will manage the parameters and invoke the kernels during forward and backward passes.
</think>

```python
# filename: linear_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 256, 'GROUP_SIZE_B': 2}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_ib, stride_ii,
    stride_wo, stride_wi,
    stride_ob, stride_oo,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute forward pass for linear layer using Triton.

    Args:
        input_ptr (tl.tensor): Pointer to input tensor of shape (B, I)
        weight_ptr (tl.tensor): Pointer to weight tensor of shape (O, I)
        output_ptr (tl.tensor): Pointer to output tensor of shape (B, O)
        B (int): Batch size dimension
        I (int): Input feature dimension
        O (int): Output feature dimension
        stride_ib (int): Input tensor batch dimension stride
        stride_ii (int): Input tensor feature dimension stride
        stride_wo (int): Weight tensor output dimension stride
        stride_wi (int): Weight tensor input dimension stride
        stride_ob (int): Output tensor batch dimension stride
        stride_oo (int): Output tensor feature dimension stride
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch super blocks

    Explanation:
        1. Create program IDs with super-block grouping
        2. Load blocks of input and weight matrices
        3. Compute matrix multiplication using dot product
        4. Store result in output tensor
    """
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    pid_b = (pid // (num_pid_o * GROUP_SIZE_B)) * GROUP_SIZE_B + (pid % GROUP_SIZE_B)
    pid_o = (pid // GROUP_SIZE_B) % num_pid_o
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_ib + offs_i[None, :] * stride_ii
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        mask_i = i * BLOCK_SIZE_I + offs_i < I
        input = tl.load(input_ptrs, mask=mask_i[None, :] & (offs_b[:, None] < B), other=0.0)
        weight = tl.load(weight_ptrs, mask=mask_i[None, :] & (offs_o[:, None] < O), other=0.0)
        acc += tl.dot(input, weight.T)
        input_ptrs += BLOCK_SIZE_I * stride_ii
        weight_ptrs += BLOCK_SIZE_I * stride_wi
    
    offs_ob = offs_b[:, None]
    offs_oo = offs_o[None, :]
    output_ptrs = output_ptr + offs_ob * stride_ob + offs_oo * stride_oo
    mask = (offs_ob < B) & (offs_oo < O)
    tl.store(output_ptrs, acc.to(tl.float16), mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 256, 'GROUP_SIZE_B': 2}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_dx_kernel(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, I, O,
    stride_dob, stride_doo,
    stride_wo, stride_wi,
    stride_dib, stride_dii,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute input gradient for linear layer backward pass.

    Args:
        d_output_ptr (tl.tensor): Pointer to output gradient tensor (B, O)
        weight_ptr (tl.tensor): Pointer to weight tensor (O, I)
        d_input_ptr (tl.tensor): Pointer to input gradient tensor (B, I)
        B (int): Batch size dimension
        I (int): Input feature dimension
        O (int): Output feature dimension
        stride_dob (int): Output gradient batch dimension stride
        stride_doo (int): Output gradient feature dimension stride
        stride_wo (int): Weight output dimension stride
        stride_wi (int): Weight input dimension stride
        stride_dib (int): Input gradient batch dimension stride
        stride_dii (int): Input gradient feature dimension stride
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch super blocks

    Explanation:
        1. Create program IDs with super-block grouping
        2. Load blocks of d_output and weight matrices
        3. Compute matrix multiplication for input gradient
        4. Store result in d_input tensor
    """
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_O)
    pid_b = (pid // (num_pid_i * GROUP_SIZE_B)) * GROUP_SIZE_B + (pid % GROUP_SIZE_B)
    pid_i = (pid // GROUP_SIZE_B) % num_pid_i
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    
    d_output_ptrs = d_output_ptr + offs_b[:, None] * stride_dob + offs_o[None, :] * stride_doo
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        mask_o = o * BLOCK_SIZE_O + offs_o < O
        d_output = tl.load(d_output_ptrs, mask=mask_o[None, :] & (offs_b[:, None] < B), other=0.0)
        weight = tl.load(weight_ptrs, mask=mask_o[:, None] & (offs_i[None, :] < I), other=0.0)
        acc += tl.dot(d_output, weight)
        d_output_ptrs += BLOCK_SIZE_O * stride_doo
        weight_ptrs += BLOCK_SIZE_O * stride_wo
    
    offs_dib = offs_b[:, None]
    offs_dii = offs_i[None, :]
    d_input_ptrs = d_input_ptr + offs_dib * stride_dib + offs_dii * stride_dii
    mask = (offs_dib < B) & (offs_dii < I)
    tl.store(d_input_ptrs, acc.to(tl.float16), mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_I': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 256, 'GROUP_SIZE_I': 2}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_dw_kernel(
    input_ptr, d_output_ptr, d_weight_ptr,
    B, I, O,
    stride_ib, stride_ii,
    stride_dob, stride_doo,
    stride_dwo, stride_dwi,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_I: tl.constexpr,
):
    """
    Compute weight gradient for linear layer backward pass.

    Args:
        input_ptr (tl.tensor): Pointer to input tensor (B, I)
        d_output_ptr (tl.tensor): Pointer to output gradient tensor (B, O)
        d_weight_ptr (tl.tensor): Pointer to weight gradient tensor (I, O)
        B (int): Batch size dimension
        I (int): Input feature dimension
        O (int): Output feature dimension
        stride_ib (int): Input batch dimension stride
        stride_ii (int): Input feature dimension stride
        stride_dob (int): Output gradient batch dimension stride
        stride_doo (int): Output gradient feature dimension stride
        stride_dwo (int): Weight gradient output dimension stride
        stride_dwi (int): Weight gradient input dimension stride
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_I (tl.constexpr): Group size for input super blocks

    Explanation:
        1. Create program IDs with super-block grouping
        2. Load blocks of input and d_output matrices
        3. Compute outer product for weight gradient
        4. Store result in d_weight tensor
    """
    pid = tl.program_id(0)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    pid_i = (pid // (num_pid_o * GROUP_SIZE_I)) * GROUP_SIZE_I + (pid % GROUP_SIZE_I)
    pid_o = (pid // GROUP_SIZE_I) % num_pid_o
    
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_b = tl.arange(0, BLOCK_SIZE_I)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_ib + offs_i[None, :] * stride_ii
    d_output_ptrs = d_output_ptr + offs_b[:, None] * stride_dob + offs_o[None, :] * stride_doo
    
    acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_O), dtype=tl.float32)
    
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_I)):
        mask_b = b * BLOCK_SIZE_I + offs_b < B
        input = tl.load(input_ptrs, mask=mask_b[:, None] & (offs_i[None, :] < I), other=0.0)
        d_output = tl.load(d_output_ptrs, mask=mask_b[:, None] & (offs_o[None, :] < O), other=0.0)
        acc += tl.dot(input.T, d_output)
        input_ptrs += BLOCK_SIZE_I * stride_ib
        d_output_ptrs += BLOCK_SIZE_I * stride_dob
    
    offs_dwi = offs_i[:, None]
    offs_dwo = offs_o[None, :]
    d_weight_ptrs = d_weight_ptr + offs_dwi * stride_dwi + offs_dwo * stride_dwo
    mask = (offs_dwi < I) & (offs_dwo < O)
    tl.store(d_weight_ptrs, acc.to(tl.float16), mask=mask)

class TritonLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        B, I = input.shape
        O, _ = weight.shape
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(O, meta['BLOCK_SIZE_O']) * meta['GROUP_SIZE_B'],)
        
        linear_forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O, _ = weight.shape
        
        d_input = torch.empty_like(input)
        d_weight = torch.zeros_like(weight)
        
        # Compute d_input
        grid_dx = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_O']) * meta['GROUP_SIZE_B'],)
        linear_backward_dx_kernel[grid_dx](
            d_output, weight, d_input,
            B, I, O,
            d_output.stride(0), d_output.stride(1),
            weight.stride(0), weight.stride(1),
            d_input.stride(0), d_input.stride(1),
        )
        
        # Compute d_weight
        grid_dw = lambda meta: (triton.cdiv(I, meta['BLOCK_SIZE_I']) * triton.cdiv(O, meta['BLOCK_SIZE_O']) * meta['GROUP_SIZE_I'],)
        linear_backward_dw_kernel[grid_dw](
            input, d_output, d_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            d_output.stride(0), d_output.stride(1),
            d_weight.stride(0), d_weight.stride(1),
        )
        
        return d_input, d_weight

class TritonLinearModule(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        torch.nn.init.xavier_normal_(self.weight)
        
    def forward(self, x):
        return TritonLinear.apply(x, self.weight)

if __name__ == "__main__":
    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    def module_generator(B, I, O, device='cuda', dtype=torch.float16):
        triton_module = TritonLinearModule(I, O, device=device, dtype=dtype)
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
To test the TritonLinear implementation, run the following command. Ensure you have CUDA-compatible GPU and required dependencies installed.

```bash
python linear_triton.py
```
</test>