Timestamp: 20250315_191635

<think>
Okay, I need to generate a Triton implementation of PyTorch's nn.Linear module, including both forward and backward passes. The filename should be linear_triton.py. The user mentioned using autotune for block sizes (BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O) in both forward and backward kernels, along with super blocks and autotune for GROUP_SIZE_* parameters. Also, the provided test code needs to be included in the __main__ block.

First, I'll start by understanding the structure of a linear layer. The forward pass is a matrix multiplication between the input and the weight matrix, possibly adding a bias. The backward pass involves computing gradients for the input and weights based on the upstream gradient.

For Triton, the key is to write efficient GPU kernels using Triton's DSL. Autotuning is essential for selecting optimal block sizes and other parameters. The @triton.autotune decorator will be used to specify configs for different block sizes and group sizes.

In the forward kernel, the input (B x I) is multiplied by the weight (O x I) to produce the output (B x O). The kernel should handle batches by partitioning the batch dimension (B) into blocks. Autotune parameters for BLOCK_SIZE_B, BLOCK_SIZE_I, and BLOCK_SIZE_O will be tested to find the best performance.

The backward pass has two parts: computing the gradient of the input (dx) and the gradient of the weights (dw). For dx, we multiply the upstream gradient (d_output, B x O) with the weight matrix (O x I), resulting in dx (B x I). For dw, we perform a matrix multiplication between the transposed input (I x B) and d_output (B x O), resulting in dw (O x I).

Each kernel will use Triton's autotune to try different block sizes and group sizes. Super blocks might refer to grouping multiple blocks to improve memory access patterns. I'll need to set up the configs with different combinations of these parameters.

Next, the TritonLinear class needs to subclass torch.nn.Module. It should initialize the weight matrix, handle the forward pass, and register the backward functions. The weight should be a parameter stored in the specified dtype and device.

In the forward method, I'll reshape the input if necessary and call the Triton kernel. For the backward method, I'll create a Backward object that computes the gradients using the Triton backward kernels.

The provided test code in the __main__ section uses B_sizes, I_sizes, O_sizes to test various dimensions. The module_generator creates instances of TritonLinear and torch.nn.Linear, and the forward/backward generators create input data. The benchmark functions from benchmark.py are used to verify correctness and measure performance.

I need to make sure that the TritonLinear module does not include a bias, as the test code initializes torch.nn.Linear with bias=False. Also, the weight matrices must be correctly initialized and transferred to the appropriate device and dtype.

Potential issues to consider: ensuring that the Triton kernels handle different block sizes correctly, especially when dimensions aren't multiples of the block sizes. Using proper memory strides and handling boundary conditions in the kernels. Also, verifying that the autotune configurations are correctly specified for both forward and backward passes.

Testing the code will involve running the provided script, which checks correctness against PyTorch's implementation and benchmarks performance. Any discrepancies in the outputs could indicate issues in the kernel implementations or parameter handling.

In summary, the steps are:

1. Implement the forward Triton kernel with autotune for block and group sizes.
2. Implement the backward kernels (dx and dw) with autotune.
3. Create the TritonLinear class that uses these kernels.
4. Include the test code as specified to run correctness and performance checks.
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
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 2}, num_warps=8),
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
    Forward pass kernel for TritonLinear layer.

    Args:
        input_ptr (tl.tensor): Input tensor of shape (B, I)
        weight_ptr (tl.tensor): Weight tensor of shape (O, I)
        output_ptr (tl.tensor): Output tensor of shape (B, O)
        B (int): Batch size dimension
        I (int): Input feature dimension
        O (int): Output feature dimension
        stride_*: Memory strides for respective dimensions
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch blocking

    Returns:
        None: Writes results to output_ptr

    Explanation:
        1. Partition batch dimension into super blocks
        2. Load blocks of input and weight matrices
        3. Compute matrix multiplication using dot product
        4. Store results in output tensor with proper blocking
    """
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    pid_group = pid // num_pid_b
    pid_b = pid % num_pid_b

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    offs_o = pid_group * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(B, I),
        strides=(stride_input_b, stride_input_i),
        offsets=(pid_b * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_I),
        order=(1, 0)
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(O, I),
        strides=(stride_weight_o, stride_weight_i),
        offsets=(pid_group * BLOCK_SIZE_O, 0),
        block_shape=(BLOCK_SIZE_O, BLOCK_SIZE_I),
        order=(1, 0)
    )

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I * GROUP_SIZE_B)):
        input = tl.load(input_block_ptr, boundary_check=(0, 1))
        weight = tl.load(weight_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(input, weight, allow_tf32=True)
        
        input_block_ptr = tl.advance(input_block_ptr, [0, BLOCK_SIZE_I])
        weight_block_ptr = tl.advance(weight_block_ptr, [0, BLOCK_SIZE_I])

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_group * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    output_ptrs = output_ptr + offs_b[:, None] * stride_output_b + offs_o[None, :] * stride_output_o
    
    tl.store(output_ptrs, accumulator, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_O': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_O': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_O': 2}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_dx_kernel(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, I, O,
    stride_dout_b, stride_dout_o,
    stride_weight_o, stride_weight_i,
    stride_din_b, stride_din_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr
):
    """
    Backward pass kernel for input gradient computation.

    Args:
        d_output_ptr (tl.tensor): Gradient of output tensor (B, O)
        weight_ptr (tl.tensor): Weight tensor (O, I)
        d_input_ptr (tl.tensor): Gradient of input tensor (B, I)
        B (int): Batch size
        I (int): Input dimension
        O (int): Output dimension
        stride_*: Memory strides
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_O (tl.constexpr): Group size for output blocking

    Returns:
        None: Writes gradients to d_input_ptr

    Explanation:
        1. Partition output dimension into super blocks
        2. Load blocks of d_output and weight matrices
        3. Compute gradient contribution using dot product
        4. Accumulate and store results in d_input tensor
    """
    pid = tl.program_id(0)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_B)
    pid_group = pid // num_pid_i
    pid_i = pid % num_pid_i

    offs_b = pid_i * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_group * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

    dout_block_ptr = tl.make_block_ptr(
        base=d_output_ptr,
        shape=(B, O),
        strides=(stride_dout_b, stride_dout_o),
        offsets=(0, pid_group * BLOCK_SIZE_O),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_O),
        order=(0, 1)
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(O, I),
        strides=(stride_weight_o, stride_weight_i),
        offsets=(pid_group * BLOCK_SIZE_O, 0),
        block_shape=(BLOCK_SIZE_O, BLOCK_SIZE_B),
        order=(1, 0)
    )

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_B), dtype=tl.float32)
    
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O * GROUP_SIZE_O)):
        dout = tl.load(dout_block_ptr, boundary_check=(0, 1))
        weight = tl.load(weight_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(dout, weight, allow_tf32=True)
        
        dout_block_ptr = tl.advance(dout_block_ptr, [0, BLOCK_SIZE_O])
        weight_block_ptr = tl.advance(weight_block_ptr, [BLOCK_SIZE_O, 0])

    offs_b = pid_i * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = tl.arange(0, BLOCK_SIZE_B)
    d_input_ptrs = d_input_ptr + offs_b[:, None] * stride_din_b + offs_i[None, :] * stride_din_i
    
    tl.store(d_input_ptrs, accumulator, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_I': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_I': 2}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_dw_kernel(
    input_ptr, d_output_ptr, d_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_dout_b, stride_dout_o,
    stride_dw_o, stride_dw_i,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_I: tl.constexpr
):
    """
    Backward pass kernel for weight gradient computation.

    Args:
        input_ptr (tl.tensor): Input tensor (B, I)
        d_output_ptr (tl.tensor): Gradient of output tensor (B, O)
        d_weight_ptr (tl.tensor): Gradient of weight tensor (O, I)
        B (int): Batch size
        I (int): Input dimension
        O (int): Output dimension
        stride_*: Memory strides
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_I (tl.constexpr): Group size for input blocking

    Returns:
        None: Writes gradients to d_weight_ptr

    Explanation:
        1. Partition input and output dimensions into blocks
        2. Load blocks of input and d_output matrices
        3. Compute outer product for weight gradient
        4. Accumulate and store results in d_weight tensor
    """
    pid = tl.program_id(0)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    pid_group = pid // num_pid_o
    pid_o = pid % num_pid_o

    offs_i = pid_group * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(B, I),
        strides=(stride_input_b, stride_input_i),
        offsets=(0, pid_group * BLOCK_SIZE_I),
        block_shape=(BLOCK_SIZE_I, BLOCK_SIZE_I),
        order=(1, 0)
    )

    dout_block_ptr = tl.make_block_ptr(
        base=d_output_ptr,
        shape=(B, O),
        strides=(stride_dout_b, stride_dout_o),
        offsets=(0, pid_o * BLOCK_SIZE_O),
        block_shape=(BLOCK_SIZE_O, BLOCK_SIZE_O),
        order=(1, 0)
    )

    accumulator = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_O), dtype=tl.float32)
    
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_I * GROUP_SIZE_I)):
        inp = tl.load(input_block_ptr, boundary_check=(0, 1))
        dout = tl.load(dout_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(inp, dout, allow_tf32=True)
        
        input_block_ptr = tl.advance(input_block_ptr, [BLOCK_SIZE_I, 0])
        dout_block_ptr = tl.advance(dout_block_ptr, [BLOCK_SIZE_O, 0])

    offs_i = pid_group * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    d_weight_ptrs = d_weight_ptr + offs_o[:, None] * stride_dw_o + offs_i[None, :] * stride_dw_i
    
    tl.store(d_weight_ptrs, accumulator, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I))


class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.backward_fn = Backward.apply

    def forward(self, x):
        if x.dim() != 2:
            x = x.view(-1, x.size(-1))
        
        B, I = x.shape
        O = self.weight.size(0)
        output = torch.empty(B, O, device=x.device, dtype=x.dtype)
        
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(O, meta['BLOCK_SIZE_O']),)
        
        linear_forward_kernel[grid](
            x, self.weight, output,
            B, I, O,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1)
        )
        
        return output

    def backward(self, d_output):
        return self.backward_fn(d_output, self)


class Backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d_output, module):
        ctx.save_for_backward(d_output, module.weight)
        return d_output

    @staticmethod
    def backward(ctx, grad_output):
        d_output, weight = ctx.saved_tensors
        input = ctx.needs_input_grad[0]
        
        B, O = d_output.shape
        I = weight.size(1)
        
        dx = dw = None
        
        if ctx.needs_input_grad[0]:
            dx = torch.empty(B, I, device=d_output.device, dtype=d_output.dtype)
            grid_dx = lambda meta: (triton.cdiv(I, meta['BLOCK_SIZE_B']) * triton.cdiv(O, meta['BLOCK_SIZE_O']),)
            linear_backward_dx_kernel[grid_dx](
                d_output, weight, dx,
                B, I, O,
                d_output.stride(0), d_output.stride(1),
                weight.stride(0), weight.stride(1),
                dx.stride(0), dx.stride(1)
            )
        
        if weight.requires_grad:
            dw = torch.zeros_like(weight)
            grid_dw = lambda meta: (triton.cdiv(I, meta['BLOCK_SIZE_I']) * triton.cdiv(O, meta['BLOCK_SIZE_O']),)
            linear_backward_dw_kernel[grid_dw](
                input, d_output, dw,
                B, I, O,
                input.stride(0), input.stride(1),
                d_output.stride(0), d_output.stride(1),
                dw.stride(0), dw.stride(1)
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
```bash
python3 linear_triton.py
```
</test>