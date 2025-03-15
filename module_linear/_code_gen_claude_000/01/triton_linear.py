# filename: triton_linear.py
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_O': 16, 'GROUP_SIZE_B': 1}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 2}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_forward_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, output_ptr,
    # Matrix dimensions
    B, I, O,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the matrix multiplication X @ W.T for the forward pass of linear layer.

    Args:
        x_ptr (tl.pointer): Pointer to the input matrix. [shape=(B, I)]
        w_ptr (tl.pointer): Pointer to the weight matrix. [shape=(O, I)]
        output_ptr (tl.pointer): Pointer to the output matrix. [shape=(B, O)]
        B (int): Batch size.
        I (int): Input dimension.
        O (int): Output dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension.

    Returns:
        None: Results are stored in output_ptr.

    Explanation:
        1. Calculate the program and block IDs using tiled layout.
        2. Load blocks of input matrix X and weight matrix W.
        3. Compute matrix multiplication X @ W.T in blocks.
        4. Accumulate results across input dimension blocks.
        5. Store the final output to output_ptr.
    """
    # Program ID
    pid_batch = tl.program_id(axis=0)
    pid_output = tl.program_id(axis=1)

    # Number of output blocks
    num_output_blocks = tl.cdiv(O, BLOCK_SIZE_O)
    # ID of the group this program belongs to
    group_id = pid_batch // GROUP_SIZE_B
    # Number of programs in this group
    num_programs_in_group = min(GROUP_SIZE_B, tl.cdiv(B, BLOCK_SIZE_B) - group_id * GROUP_SIZE_B)
    # This program's ID within the group
    program_id_in_group = pid_batch % GROUP_SIZE_B
    # Batch start offset for this program
    batch_start = group_id * GROUP_SIZE_B * BLOCK_SIZE_B + program_id_in_group * BLOCK_SIZE_B
    # Output block ID for this program
    output_block_id = pid_output
    # Output block start offset
    output_start = output_block_id * BLOCK_SIZE_O

    # Block indices
    batch_index = batch_start + tl.arange(0, BLOCK_SIZE_B)
    output_index = output_start + tl.arange(0, BLOCK_SIZE_O)

    # Create mask for valid indices
    batch_mask = batch_index < B
    output_mask = output_index < O

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)

    # Loop over I dimension blocks
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        # Compute indices for current I block
        i_mask = (i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)) < I

        # Load input block (B, I)
        x_block_ptr = x_ptr + batch_index[:, None] * I + i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)[None, :]
        x_block = tl.load(x_block_ptr, mask=batch_mask[:, None] & i_mask[None, :], other=0.0)

        # Load weight block (O, I)
        w_block_ptr = w_ptr + output_index[:, None] * I + i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)[None, :]
        w_block = tl.load(w_block_ptr, mask=output_mask[:, None] & i_mask[None, :], other=0.0)

        # Matrix multiplication for this block
        accumulator += tl.dot(x_block, tl.trans(w_block))

    # Store output block
    output_block_ptr = output_ptr + batch_index[:, None] * O + output_index[None, :]
    tl.store(output_block_ptr, accumulator, mask=batch_mask[:, None] & output_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 16, 'GROUP_SIZE_B': 1}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 2}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_dx_kernel(
    # Pointers to matrices
    dy_ptr, w_ptr, dx_ptr,
    # Matrix dimensions
    B, I, O,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the gradient with respect to input: dx = dy @ W.

    Args:
        dy_ptr (tl.pointer): Pointer to the gradient of outputs. [shape=(B, O)]
        w_ptr (tl.pointer): Pointer to the weight matrix. [shape=(O, I)]
        dx_ptr (tl.pointer): Pointer to the gradient of inputs. [shape=(B, I)]
        B (int): Batch size.
        I (int): Input dimension.
        O (int): Output dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension.
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension.

    Returns:
        None: Results are stored in dx_ptr.

    Explanation:
        1. Calculate the program and block IDs using tiled layout.
        2. Load blocks of gradient dy and weight matrix W.
        3. Compute matrix multiplication dy @ W in blocks.
        4. Accumulate results across output dimension blocks.
        5. Store the final gradient to dx_ptr.
    """
    # Program ID
    pid_batch = tl.program_id(axis=0)
    pid_input = tl.program_id(axis=1)

    # Number of input blocks
    num_input_blocks = tl.cdiv(I, BLOCK_SIZE_I)
    # ID of the group this program belongs to
    group_id = pid_batch // GROUP_SIZE_B
    # Number of programs in this group
    num_programs_in_group = min(GROUP_SIZE_B, tl.cdiv(B, BLOCK_SIZE_B) - group_id * GROUP_SIZE_B)
    # This program's ID within the group
    program_id_in_group = pid_batch % GROUP_SIZE_B
    # Batch start offset for this program
    batch_start = group_id * GROUP_SIZE_B * BLOCK_SIZE_B + program_id_in_group * BLOCK_SIZE_B
    # Input block ID for this program
    input_block_id = pid_input
    # Input block start offset
    input_start = input_block_id * BLOCK_SIZE_I

    # Block indices
    batch_index = batch_start + tl.arange(0, BLOCK_SIZE_B)
    input_index = input_start + tl.arange(0, BLOCK_SIZE_I)

    # Create mask for valid indices
    batch_mask = batch_index < B
    input_mask = input_index < I

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)

    # Loop over O dimension blocks
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        # Compute indices for current O block
        o_start = o * BLOCK_SIZE_O
        o_index = o_start + tl.arange(0, BLOCK_SIZE_O)
        o_mask = o_index < O

        # Load gradient of output block (B, O)
        dy_block_ptr = dy_ptr + batch_index[:, None] * O + o_index[None, :]
        dy_block = tl.load(dy_block_ptr, mask=batch_mask[:, None] & o_mask[None, :], other=0.0)

        # Load weight block (O, I)
        w_block_ptr = w_ptr + o_index[:, None] * I + input_index[None, :]
        w_block = tl.load(w_block_ptr, mask=o_mask[:, None] & input_mask[None, :], other=0.0)

        # Matrix multiplication for this block: dy @ w
        accumulator += tl.dot(dy_block, w_block)

    # Store output block (gradient of input)
    dx_block_ptr = dx_ptr + batch_index[:, None] * I + input_index[None, :]
    tl.store(dx_block_ptr, accumulator, mask=batch_mask[:, None] & input_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_B': 16, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 2}),
        triton.Config({'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 4}),
        triton.Config({'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_dw_kernel(
    # Pointers to matrices
    x_ptr, dy_ptr, dw_ptr,
    # Matrix dimensions
    B, I, O,
    # Meta-parameters
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute the gradient with respect to weights: dW = x.T @ dy.

    Args:
        x_ptr (tl.pointer): Pointer to the input matrix. [shape=(B, I)]
        dy_ptr (tl.pointer): Pointer to the gradient of outputs. [shape=(B, O)]
        dw_ptr (tl.pointer): Pointer to the gradient of weights. [shape=(O, I)]
        B (int): Batch size.
        I (int): Input dimension.
        O (int): Output dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        GROUP_SIZE_O (tl.constexpr): Group size for output dimension.

    Returns:
        None: Results are stored in dw_ptr.

    Explanation:
        1. Calculate the program and block IDs using tiled layout.
        2. Load blocks of input x and gradient dy.
        3. Compute matrix multiplication x.T @ dy in blocks.
        4. Accumulate results across batch dimension blocks.
        5. Store the final gradient to dw_ptr.
    """
    # Program ID
    pid_output = tl.program_id(axis=0)
    pid_input = tl.program_id(axis=1)

    # ID of the group this program belongs to
    group_id = pid_output // GROUP_SIZE_O
    # Number of programs in this group
    num_programs_in_group = min(GROUP_SIZE_O, tl.cdiv(O, BLOCK_SIZE_O) - group_id * GROUP_SIZE_O)
    # This program's ID within the group
    program_id_in_group = pid_output % GROUP_SIZE_O
    # Output start offset for this program
    output_start = group_id * GROUP_SIZE_O * BLOCK_SIZE_O + program_id_in_group * BLOCK_SIZE_O
    # Input block ID for this program
    input_block_id = pid_input
    # Input block start offset
    input_start = input_block_id * BLOCK_SIZE_I

    # Block indices
    output_index = output_start + tl.arange(0, BLOCK_SIZE_O)
    input_index = input_start + tl.arange(0, BLOCK_SIZE_I)

    # Create mask for valid indices
    output_mask = output_index < O
    input_mask = input_index < I

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)

    # Loop over B dimension blocks
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        # Compute indices for current B block
        b_start = b * BLOCK_SIZE_B
        b_index = b_start + tl.arange(0, BLOCK_SIZE_B)
        b_mask = b_index < B

        # Load input block (B, I)
        x_block_ptr = x_ptr + b_index[:, None] * I + input_index[None, :]
        x_block = tl.load(x_block_ptr, mask=b_mask[:, None] & input_mask[None, :], other=0.0)

        # Load gradient of output block (B, O)
        dy_block_ptr = dy_ptr + b_index[:, None] * O + output_index[None, :]
        dy_block = tl.load(dy_block_ptr, mask=b_mask[:, None] & output_mask[None, :], other=0.0)

        # Matrix multiplication for this block: x.T @ dy
        accumulator += tl.dot(tl.trans(dy_block), x_block)

    # Store output block (gradient of weights)
    dw_block_ptr = dw_ptr + output_index[:, None] * I + input_index[None, :]
    tl.store(dw_block_ptr, accumulator, mask=output_mask[:, None] & input_mask[None, :])


class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """
        Forward pass for TritonLinear layer.

        Args:
            ctx (Context): Context object to save tensors for backward.
            x (torch.Tensor): Input tensor. [shape=(B, I)]
            weight (torch.Tensor): Weight tensor. [shape=(O, I)]

        Returns:
            torch.Tensor: Output tensor. [shape=(B, O)]

        Explanation:
            1. Save input and weights for backward pass.
            2. Get dimensions from input and weight tensors.
            3. Allocate output tensor.
            4. Run Triton kernel for forward pass.
            5. Return the computed output.
        """
        ctx.save_for_backward(x, weight)

        B, I = x.shape
        O = weight.shape[0]

        # Allocate output
        output = torch.empty((B, O), device=x.device, dtype=x.dtype)

        # Launch kernel
        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']) * meta['GROUP_SIZE_B'],
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
        )

        linear_forward_kernel[grid](
            x, weight, output,
            B, I, O,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for TritonLinear layer.

        Args:
            ctx (Context): Context object with saved tensors.
            grad_output (torch.Tensor): Gradient of the loss with respect to output. [shape=(B, O)]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Gradients for input and weight.

        Explanation:
            1. Retrieve saved tensors from context.
            2. Get dimensions from tensors.
            3. Allocate tensors for gradients.
            4. Run Triton kernels for computing gradients:
               a. First kernel: Computes gradient with respect to input (dx).
               b. Second kernel: Computes gradient with respect to weights (dw).
            5. Return both gradients.
        """
        x, weight = ctx.saved_tensors
        B, I = x.shape
        O = weight.shape[0]

        # Allocate gradients
        grad_input = torch.empty_like(x)
        grad_weight = torch.empty_like(weight)

        # Launch kernels
        dx_grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']) * meta['GROUP_SIZE_B'],
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )

        dw_grid = lambda meta: (
            triton.cdiv(O, meta['BLOCK_SIZE_O']) * meta['GROUP_SIZE_O'],
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )

        linear_backward_dx_kernel[dx_grid](
            grad_output, weight, grad_input,
            B, I, O,
        )

        linear_backward_dw_kernel[dw_grid](
            x, grad_output, grad_weight,
            B, I, O,
        )

        return grad_input, grad_weight


class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        """
        Initialize TritonLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            device (str): Device to store parameters, default='cuda'.
            dtype (torch.dtype): Data type for parameters, default=torch.float16.

        Returns:
            None

        Explanation:
            1. Initialize the layer with required parameters.
            2. Create weight parameter with specified dimensions, device, and dtype.
            3. Reset parameters with proper initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters using Kaiming initialization.

        Args:
            None

        Returns:
            None

        Explanation:
            1. Apply Kaiming uniform initialization to the weights.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        """
        Forward pass through the linear layer.

        Args:
            x (torch.Tensor): Input tensor. [shape=(B, I)]

        Returns:
            torch.Tensor: Output tensor. [shape=(B, O)]

        Explanation:
            1. Check if input dtype matches weight dtype.
            2. Call the custom autograd function to perform the forward operation.
        """
        # Ensure types match
        assert x.dtype == self.weight.dtype, f"Input dtype ({x.dtype}) doesn't match weight dtype ({self.weight.dtype})"

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
