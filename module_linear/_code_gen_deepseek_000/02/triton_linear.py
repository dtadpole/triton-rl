# filename: triton_linear.py
import torch
import triton
import triton.language as tl

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Description: Performs forward pass of linear layer using Triton kernels.

        Args:
            ctx (torch.autograd.function.BackwardCFunction): Context for saving tensors.
            input (torch.Tensor): Input tensor of shape (B, I).
            weight (torch.Tensor): Weight tensor of shape (O, I).

        Returns:
            torch.Tensor: Output tensor of shape (B, O).

        Explanation:
            1. Save input and weight for backward pass.
            2. Initialize output tensor.
            3. Launch Triton kernel with autotuned block sizes.
        """
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O = weight.shape[0]
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)

        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
        )

        triton_linear_forward[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Description: Performs backward pass of linear layer using Triton kernels.

        Args:
            ctx (torch.autograd.function.BackwardCFunction): Context with saved tensors.
            grad_output (torch.Tensor): Gradient of output tensor of shape (B, O).

        Returns:
            tuple: (grad_input, grad_weight) tensors of shapes (B, I) and (O, I).

        Explanation:
            1. Retrieve saved tensors from forward pass.
            2. Initialize gradient tensors.
            3. Launch input gradient kernel with autotuned block sizes.
            4. Launch weight gradient kernel with autotuned block sizes.
        """
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O = weight.shape[0]

        grad_input = torch.empty_like(input)
        grad_weight = torch.zeros_like(weight)

        # Compute input gradient
        grid_input = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        triton_linear_backward_input[grid_input](
            grad_output, weight, grad_input,
            B, O, I,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1),
        )

        # Compute weight gradient
        grid_weight = lambda meta: (
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        triton_linear_backward_weight[grid_weight](
            grad_output, input, grad_weight,
            B, O, I,
            grad_output.stride(0), grad_output.stride(1),
            input.stride(0), input.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
        )

        return grad_input, grad_weight

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def triton_linear_forward(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    input_row_stride, input_col_stride,
    weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
):
    """
    Description: Triton kernel for linear layer forward pass.

    Args:
        input_ptr (tensor): Input tensor of shape (B, I).
        weight_ptr (tensor): Weight tensor of shape (O, I).
        output_ptr (tensor): Output tensor of shape (B, O).
        B (int): Batch size dimension.
        I (int): Input feature dimension.
        O (int): Output feature dimension.
        input_row_stride (int): Input tensor row stride.
        input_col_stride (int): Input tensor column stride.
        weight_row_stride (int): Weight tensor row stride.
        weight_col_stride (int): Weight tensor column stride.
        output_row_stride (int): Output tensor row stride.
        output_col_stride (int): Output tensor column stride.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension.

    Returns:
        None: Output is stored in output_ptr.

    Explanation:
        1. Calculate block indices for B and O dimensions.
        2. Initialize accumulator to zero.
        3. Loop over I dimension in BLOCK_SIZE_I increments.
        4. Load input and weight tiles from global memory.
        5. Compute matrix multiplication using tensor cores.
        6. Store accumulated results to output.
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)

    input_ptrs = input_ptr + offs_b[:, None] * input_row_stride + offs_i[None, :] * input_col_stride
    weight_ptrs = weight_ptr + offs_o[:, None] * weight_row_stride + offs_i[None, :] * weight_col_stride

    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)

    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        mask_i = i * BLOCK_SIZE_I + offs_i < I
        input = tl.load(input_ptrs, mask=mask_i[None, :] & (offs_b[:, None] < B), other=0.0)
        weight = tl.load(weight_ptrs, mask=mask_i[None, :] & (offs_o[:, None] < O), other=0.0)
        acc += tl.dot(input, tl.trans(weight))
        input_ptrs += BLOCK_SIZE_I * input_col_stride
        weight_ptrs += BLOCK_SIZE_I * weight_col_stride

    output_ptrs = output_ptr + offs_b[:, None] * output_row_stride + offs_o[None, :] * output_col_stride
    tl.store(output_ptrs, acc.to(tl.float16), mask=(offs_b[:, None] < B) & (offs_o[None, :] < O))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64}, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def triton_linear_backward_input(
    grad_output_ptr, weight_ptr, grad_input_ptr,
    B, O, I,
    grad_output_row_stride, grad_output_col_stride,
    weight_row_stride, weight_col_stride,
    grad_input_row_stride, grad_input_col_stride,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
):
    """
    Description: Triton kernel for computing input gradients.

    Args:
        grad_output_ptr (tensor): Output gradient tensor of shape (B, O).
        weight_ptr (tensor): Weight tensor of shape (O, I).
        grad_input_ptr (tensor): Input gradient tensor of shape (B, I).
        B (int): Batch size dimension.
        O (int): Output feature dimension.
        I (int): Input feature dimension.
        grad_output_row_stride (int): Output gradient row stride.
        grad_output_col_stride (int): Output gradient column stride.
        weight_row_stride (int): Weight tensor row stride.
        weight_col_stride (int): Weight tensor column stride.
        grad_input_row_stride (int): Input gradient row stride.
        grad_input_col_stride (int): Input gradient column stride.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension.

    Returns:
        None: Input gradient is stored in grad_input_ptr.

    Explanation:
        1. Calculate block indices for B and I dimensions.
        2. Initialize accumulator to zero.
        3. Loop over O dimension in blocks.
        4. Load gradient output and weight tiles.
        5. Compute matrix multiplication and accumulate.
        6. Store accumulated results to input gradient.
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = tl.arange(0, BLOCK_SIZE_O)

    grad_output_ptrs = grad_output_ptr + offs_b[:, None] * grad_output_row_stride + offs_o[None, :] * grad_output_col_stride
    weight_ptrs = weight_ptr + offs_o[:, None] * weight_row_stride + offs_i[None, :] * weight_col_stride

    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)

    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        mask_o = o * BLOCK_SIZE_O + offs_o < O
        grad_output = tl.load(grad_output_ptrs, mask=mask_o[None, :] & (offs_b[:, None] < B), other=0.0)
        weight = tl.load(weight_ptrs, mask=mask_o[:, None] & (offs_i[None, :] < I), other=0.0)
        acc += tl.dot(grad_output, weight)
        grad_output_ptrs += BLOCK_SIZE_O * grad_output_col_stride
        weight_ptrs += BLOCK_SIZE_O * weight_row_stride

    grad_input_ptrs = grad_input_ptr + offs_b[:, None] * grad_input_row_stride + offs_i[None, :] * grad_input_col_stride
    tl.store(grad_input_ptrs, acc.to(tl.float16), mask=(offs_b[:, None] < B) & (offs_i[None, :] < I))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 64}, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def triton_linear_backward_weight(
    grad_output_ptr, input_ptr, grad_weight_ptr,
    B, O, I,
    grad_output_row_stride, grad_output_col_stride,
    input_row_stride, input_col_stride,
    grad_weight_row_stride, grad_weight_col_stride,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    """
    Description: Triton kernel for computing weight gradients.

    Args:
        grad_output_ptr (tensor): Output gradient tensor of shape (B, O).
        input_ptr (tensor): Input tensor of shape (B, I).
        grad_weight_ptr (tensor): Weight gradient tensor of shape (O, I).
        B (int): Batch size dimension.
        O (int): Output feature dimension.
        I (int): Input feature dimension.
        grad_output_row_stride (int): Output gradient row stride.
        grad_output_col_stride (int): Output gradient column stride.
        input_row_stride (int): Input tensor row stride.
        input_col_stride (int): Input tensor column stride.
        grad_weight_row_stride (int): Weight gradient row stride.
        grad_weight_col_stride (int): Weight gradient column stride.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension.

    Returns:
        None: Weight gradient is stored in grad_weight_ptr.

    Explanation:
        1. Calculate block indices for O and I dimensions.
        2. Initialize accumulator to zero.
        3. Loop over B dimension in blocks.
        4. Load gradient output and input tiles.
        5. Compute outer product and accumulate.
        6. Store accumulated results to weight gradient.
    """
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = tl.arange(0, BLOCK_SIZE_B)

    grad_output_ptrs = grad_output_ptr + offs_b[:, None] * grad_output_row_stride + offs_o[None, :] * grad_output_col_stride
    input_ptrs = input_ptr + offs_b[:, None] * input_row_stride + offs_i[None, :] * input_col_stride

    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)

    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        mask_b = b * BLOCK_SIZE_B + offs_b < B
        grad_output = tl.load(grad_output_ptrs, mask=mask_b[:, None] & (offs_o[None, :] < O), other=0.0)
        input = tl.load(input_ptrs, mask=mask_b[:, None] & (offs_i[None, :] < I), other=0.0)
        acc += tl.dot(tl.trans(grad_output), input)
        grad_output_ptrs += BLOCK_SIZE_B * grad_output_row_stride
        input_ptrs += BLOCK_SIZE_B * input_row_stride

    grad_weight_ptrs = grad_weight_ptr + offs_o[:, None] * grad_weight_row_stride + offs_i[None, :] * grad_weight_col_stride
    tl.store(grad_weight_ptrs, acc.to(tl.float16), mask=(offs_o[:, None] < O) & (offs_i[None, :] < I))

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
