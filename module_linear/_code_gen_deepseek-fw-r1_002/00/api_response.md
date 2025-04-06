Timestamp: 20250406_055337

<think>
Alright, I need to create a Triton module for PyTorch's nn.Linear, implementing both forward and backward passes. The user specified using autotune for block sizes and super blocks. Let me break this down step by step.

First, the TritonLinear class should mirror the functionality of torch.nn.Linear. That means it should have a weight parameter, handle input and output dimensions, and manage device and dtype. Since the user mentioned no bias, I'll omit that.

For the forward pass, the matrix multiplication is the key part. Using Triton's autotune functionality will help optimize the block sizes. The user mentioned BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O, which correspond to batch, input, and output dimensions. Autotuning these will require defining multiple configs and letting Triton select the best one.

The forward kernel needs to load the input and weight tiles into shared memory, compute the dot product, and store the result. Using tl.dot for the computation and specifying block sizes with triton.autotune will be essential. I'll set up the autotune configs with different combinations of block sizes and group sizes as per the problem statement.

Next, the backward pass involves computing gradients for both the input (dx) and the weight (dw). The backward kernel for the input will perform a matrix multiplication between the gradient of the output (d_output) and the weight matrix. The kernel for the weight gradient will multiply the input transpose with d_output. Both of these will also need autotuned block sizes.

In the backward method of TritonLinear, I'll need to define two separate kernels: one for computing dx and another for dw. Each kernel will have its own autotune configurations. Using @triton.autotune for these kernels with the specified BLOCK_SIZE variables and GROUP_SIZE parameters is necessary.

Testing the module requires integrating with the provided benchmark code. The module_generator function should create instances of TritonLinear and torch.nn.Linear, ensuring they're on the correct device and dtype. The forward and backward generators prepare the inputs and gradients as specified.

I need to make sure that the TritonLinear's weight is initialized correctly, similar to PyTorch's initialization. Also, during the backward pass, the gradients for the weight should be accumulated properly, which might require transposing the input matrix when computing dw.

Potential issues might include incorrect block sizes leading to mismatched dimensions or inefficient memory access. Testing with different sizes (256, 1024, 4096) will help verify correctness and performance. Also, ensuring that the autotune configurations cover a sufficient range of possibilities is crucial for optimal performance.

Another consideration is the memory layout. Triton operates on contiguous tensors, so ensuring that inputs and weights are properly transposed or contiguous is important. For example, in the forward pass, the weight matrix is stored as (O, I), so accessing it in a block-wise manner should be efficient.

Finally, integrating all parts into the linear_triton.py file, making sure the kernels are correctly annotated with their descriptions, arguments, and explanation steps as per the documentation requirements. Each function's docstring should detail the steps, arguments, and return values clearly.
</think>

```python
# filename: linear_triton.py
import torch
import triton
import triton.language as tl

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return linear_forward_triton(x, weight)

    @staticmethod
    def backward(ctx, d_output):
        x, weight = ctx.saved_tensors
        dx, dw = linear_backward_triton(d_output, x, weight)
        return dx, dw

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 1024, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64}, num_warps=4),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_forward_kernel(
    x_ptr, w_ptr, out_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_wb, stride_wi,
    stride_outb, stride_outo,
    GROUP_SIZE_B: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
):
    """
    Compute forward pass for linear layer using matrix multiplication.

    Args:
        x_ptr (triton.Pointer): Input tensor pointer (shape: [B, I])
        w_ptr (triton.Pointer): Weight tensor pointer (shape: [O, I])
        out_ptr (triton.Pointer): Output tensor pointer (shape: [B, O])
        B (int): Batch size dimension
        I (int): Input feature dimension
        O (int): Output feature dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension tiling
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output feature dimension

    Explanation:
        1. Partition batch and output dimensions into super blocks
        2. Load blocks of input and weight matrices into shared memory
        3. Compute partial dot products using tensor cores
        4. Accumulate results across blocks
        5. Store final output blocks
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)

    x_ptrs = x_ptr + (offs_b[:, None]) * stride_xb + (offs_i[None, :]) * stride_xi
    w_ptrs = w_ptr + (offs_o[:, None]) * stride_wb + (offs_i[None, :]) * stride_wi

    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)

    for i in range(0, I, BLOCK_SIZE_I * GROUP_SIZE_B):
        x = tl.load(x_ptrs, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I - i), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I - i), other=0.0)
        acc += tl.dot(x, w.T)
        x_ptrs += BLOCK_SIZE_I * GROUP_SIZE_B * stride_xi
        w_ptrs += BLOCK_SIZE_I * GROUP_SIZE_B * stride_wi

    offs_out_b = offs_b[:, None]
    offs_out_o = offs_o[None, :]
    out_ptrs = out_ptr + offs_out_b * stride_outb + offs_out_o * stride_outo
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_out_b < B) & (offs_out_o < O))

def linear_forward_triton(x, weight):
    B, I = x.shape
    O, _ = weight.shape
    out = torch.empty((B, O), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(B, meta['BLOCK_SIZE_B']),
        triton.cdiv(O, meta['BLOCK_SIZE_O']),
    )
    
    linear_forward_kernel[grid](
        x, weight, out,
        B, I, O,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE_B=4
    )
    return out

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 128}, num_warps=4),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_dx_kernel(
    d_output_ptr, w_ptr, dx_ptr,
    B, I, O,
    stride_dob, stride_doo,
    stride_wb, stride_wi,
    stride_dxb, stride_dxi,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute input gradient (dx) using matrix multiplication between d_output and weight matrix.

    Args:
        d_output_ptr (triton.Pointer): Gradient of output pointer (shape: [B, O])
        w_ptr (triton.Pointer): Weight tensor pointer (shape: [O, I])
        dx_ptr (triton.Pointer): Input gradient pointer (shape: [B, I])
        B (int): Batch size dimension
        I (int): Input feature dimension
        O (int): Output feature dimension
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output feature dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension tiling

    Explanation:
        1. Partition batch and input dimensions into blocks
        2. Load blocks of gradient and weight matrices
        3. Compute partial products for input gradient
        4. Accumulate and store results
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * GROUP_SIZE_B + tl.arange(0, GROUP_SIZE_B)
    offs_o = tl.arange(0, BLOCK_SIZE_O)

    acc = tl.zeros((BLOCK_SIZE_B, GROUP_SIZE_B), dtype=tl.float32)

    for o in range(0, O, BLOCK_SIZE_O):
        do = tl.load(d_output_ptr + offs_b[:, None] * stride_dob + (o + offs_o)[None, :] * stride_doo,
                    mask=(offs_b[:, None] < B) & ((o + offs_o)[None, :] < O), other=0.0)
        w = tl.load(w_ptr + (o + offs_o)[:, None] * stride_wb + offs_i[None, :] * stride_wi,
                   mask=(o + offs_o)[:, None] < O, other=0.0)
        acc += tl.dot(do, w)

    dx_ptrs = dx_ptr + offs_b[:, None] * stride_dxb + offs_i[None, :] * stride_dxi
    tl.store(dx_ptrs, acc.to(tl.float16), mask=(offs_b[:, None] < B) & (offs_i[None, :] < I))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 128}, num_warps=4),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_dw_kernel(
    x_ptr, d_output_ptr, dw_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_dob, stride_doo,
    stride_dwb, stride_dwi,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute weight gradient (dw) using matrix multiplication between input and d_output.

    Args:
        x_ptr (triton.Pointer): Input tensor pointer (shape: [B, I])
        d_output_ptr (triton.Pointer): Gradient of output pointer (shape: [B, O])
        dw_ptr (triton.Pointer): Weight gradient pointer (shape: [O, I])
        B (int): Batch size dimension
        I (int): Input feature dimension
        O (int): Output feature dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output feature dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension tiling

    Explanation:
        1. Partition input and output dimensions into blocks
        2. Load blocks of input and gradient matrices
        3. Compute outer product for weight gradient
        4. Accumulate and store results
    """
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = tl.arange(0, GROUP_SIZE_B)

    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)

    x_ptrs = x_ptr + offs_b[None, :] * stride_xb + offs_i[:, None] * stride_xi
    do_ptrs = d_output_ptr + offs_b[None, :] * stride_dob + offs_o[:, None] * stride_doo

    for b in range(0, B, GROUP_SIZE_B):
        x = tl.load(x_ptrs, mask=(offs_b[None, :] < B - b) & (offs_i[:, None] < I), other=0.0)
        do = tl.load(do_ptrs, mask=(offs_b[None, :] < B - b) & (offs_o[:, None] < O), other=0.0)
        acc += tl.dot(do, x)
        x_ptrs += GROUP_SIZE_B * stride_xb
        do_ptrs += GROUP_SIZE_B * stride_dob

    dw_ptrs = dw_ptr + offs_o[:, None] * stride_dwb + offs_i[None, :] * stride_dwi
    tl.store(dw_ptrs, acc.to(tl.float16), mask=(offs_o[:, None] < O) & (offs_i[None, :] < I))

def linear_backward_triton(d_output, x, weight):
    B, I = x.shape
    O = weight.shape[0]
    
    dx = torch.empty_like(x)
    grid_dx = lambda meta: (
        triton.cdiv(B, meta['BLOCK_SIZE_B']),
        triton.cdiv(I, meta['GROUP_SIZE_B']),
    )
    linear_backward_dx_kernel[grid_dx](
        d_output, weight.T.contiguous(), dx,
        B, I, O,
        d_output.stride(0), d_output.stride(1),
        weight.stride(1), weight.stride(0),
        dx.stride(0), dx.stride(1),
        GROUP_SIZE_B=4
    )

    dw = torch.zeros_like(weight)
    grid_dw = lambda meta: (
        triton.cdiv(O, meta['BLOCK_SIZE_O']),
        triton.cdiv(I, meta['BLOCK_SIZE_I']),
    )
    linear_backward_dw_kernel[grid_dw](
        x.contiguous(), d_output.contiguous(), dw,
        B, I, O,
        x.stride(0), x.stride(1),
        d_output.stride(0), d_output.stride(1),
        dw.stride(0), dw.stride(1),
        GROUP_SIZE_B=4
    )

    return dx, dw

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
To test the implementation, run the following command. Ensure you have CUDA-compatible GPU and required dependencies installed.

```bash
python linear_triton.py
```
</test>