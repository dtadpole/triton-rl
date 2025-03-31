# filename: linear_triton.py
import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_O': 16, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_I': 2, 'GROUP_SIZE_O': 2}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_I': 4, 'GROUP_SIZE_O': 4}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_forward_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, y_ptr,
    # Matrix dimensions
    B, I, O,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute the forward pass of a linear layer: y = x @ W.T

    Args:
        x_ptr (ptr): Pointer to the input tensor. [B, I]
        w_ptr (ptr): Pointer to the weight tensor. [O, I]
        y_ptr (ptr): Pointer to the output tensor. [B, O]
        B (int): Batch size.
        I (int): Input feature dimension.
        O (int): Output feature dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for output feature dimension.
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension.
        GROUP_SIZE_I (tl.constexpr): Group size for input feature dimension.
        GROUP_SIZE_O (tl.constexpr): Group size for output feature dimension.

    Returns:
        None: Result is written to y_ptr.

    Explanation:
        1. Calculate block indices for the current thread.
        2. Create masks to handle boundary conditions.
        3. Initialize an accumulator for matrix multiplication.
        4. Iterate over input dimension in blocks.
        5. Load input and weight blocks with boundary checks.
        6. Perform matrix multiplication using tl.dot.
        7. Store result to output tensor.
    """
    # Compute the indices for each thread in the grid
    b_idx = tl.program_id(0) * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    o_idx = tl.program_id(1) * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    # Create masks to handle boundary conditions
    b_mask = b_idx < B
    o_mask = o_idx < O
    
    # Combine masks for output
    mask = b_mask[:, None] & o_mask[None, :]
    
    # Initialize accumulator for matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    # Iterate over the input dimension in blocks
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        i_idx = i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
        i_mask = i_idx < I
        
        # Load blocks from input and weight matrices
        x_block = tl.load(x_ptr + b_idx[:, None] * I + i_idx[None, :], 
                         mask=b_mask[:, None] & i_mask[None, :], other=0.0)
        w_block = tl.load(w_ptr + o_idx[:, None] * I + i_idx[None, :], 
                         mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        
        # Transpose weight block for matrix multiplication
        w_block_t = tl.trans(w_block)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(x_block, w_block_t, allow_tf32=False)
    
    # Store the result in the output matrix
    tl.store(y_ptr + b_idx[:, None] * O + o_idx[None, :], acc, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_O': 16, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_I': 2, 'GROUP_SIZE_O': 2}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_I': 4, 'GROUP_SIZE_O': 4}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_input_kernel(
    # Pointers to matrices
    dy_ptr, w_ptr, dx_ptr,
    # Matrix dimensions
    B, I, O,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute the gradient with respect to input: dx = dy @ W

    Args:
        dy_ptr (ptr): Pointer to the output gradient tensor. [B, O]
        w_ptr (ptr): Pointer to the weight tensor. [O, I]
        dx_ptr (ptr): Pointer to the input gradient tensor. [B, I]
        B (int): Batch size.
        I (int): Input feature dimension.
        O (int): Output feature dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for output feature dimension.
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension.
        GROUP_SIZE_I (tl.constexpr): Group size for input feature dimension.
        GROUP_SIZE_O (tl.constexpr): Group size for output feature dimension.

    Returns:
        None: Result is written to dx_ptr.

    Explanation:
        1. Calculate block indices for the current thread.
        2. Create masks to handle boundary conditions.
        3. Initialize an accumulator for matrix multiplication.
        4. Iterate over output dimension in blocks.
        5. Load gradient and weight blocks with boundary checks.
        6. Perform matrix multiplication to compute input gradients.
        7. Store result to dx_ptr.
    """
    # Compute the indices for each thread in the grid
    b_idx = tl.program_id(0) * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    i_idx = tl.program_id(1) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Create masks to handle boundary conditions
    b_mask = b_idx < B
    i_mask = i_idx < I
    
    # Combine masks for input gradient
    mask = b_mask[:, None] & i_mask[None, :]
    
    # Initialize accumulator for matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over the output dimension in blocks
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        o_idx = o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
        o_mask = o_idx < O
        
        # Load blocks from gradient and weight matrices
        dy_block = tl.load(dy_ptr + b_idx[:, None] * O + o_idx[None, :], 
                          mask=b_mask[:, None] & o_mask[None, :], other=0.0)
        w_block = tl.load(w_ptr + o_idx[:, None] * I + i_idx[None, :], 
                         mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        
        # Perform matrix multiplication for this block: dx = dy @ W
        acc += tl.dot(dy_block, w_block, allow_tf32=False)
    
    # Store the result in the gradient with respect to input matrix
    tl.store(dx_ptr + b_idx[:, None] * I + i_idx[None, :], acc, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_B': 16, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_B': 1}),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 2, 'GROUP_SIZE_I': 2, 'GROUP_SIZE_B': 2}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 4, 'GROUP_SIZE_I': 4, 'GROUP_SIZE_B': 4}),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_weight_kernel(
    # Pointers to matrices
    x_ptr, dy_ptr, dw_ptr,
    # Matrix dimensions
    B, I, O,
    # Meta-parameters
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the gradient with respect to weights: dw = dy.T @ x

    Args:
        x_ptr (ptr): Pointer to the input tensor. [B, I]
        dy_ptr (ptr): Pointer to the output gradient tensor. [B, O]
        dw_ptr (ptr): Pointer to the weight gradient tensor. [O, I]
        B (int): Batch size.
        I (int): Input feature dimension.
        O (int): Output feature dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for output feature dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        GROUP_SIZE_O (tl.constexpr): Group size for output feature dimension.
        GROUP_SIZE_I (tl.constexpr): Group size for input feature dimension.
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension.

    Returns:
        None: Result is written to dw_ptr.

    Explanation:
        1. Calculate block indices for the current thread.
        2. Create masks to handle boundary conditions.
        3. Initialize an accumulator for matrix multiplication.
        4. Iterate over batch dimension in blocks.
        5. Load input and gradient blocks with boundary checks.
        6. Transpose input block and perform matrix multiplication.
        7. Store result to weight gradient tensor.
    """
    # Compute the indices for each thread in the grid
    o_idx = tl.program_id(0) * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    i_idx = tl.program_id(1) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Create masks to handle boundary conditions
    o_mask = o_idx < O
    i_mask = i_idx < I
    
    # Combine masks for weight gradient
    mask = o_mask[:, None] & i_mask[None, :]
    
    # Initialize accumulator for matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over the batch dimension in blocks
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        b_idx = b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        b_mask = b_idx < B
        
        # Load blocks from input and gradient matrices
        x_block = tl.load(x_ptr + b_idx[:, None] * I + i_idx[None, :], 
                         mask=b_mask[:, None] & i_mask[None, :], other=0.0)
        dy_block = tl.load(dy_ptr + b_idx[:, None] * O + o_idx[None, :], 
                          mask=b_mask[:, None] & o_mask[None, :], other=0.0)
        
        # Transpose blocks for matrix multiplication: dw = dy.T @ x
        x_block_t = tl.trans(x_block)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(dy_block, x_block_t, allow_tf32=False)
    
    # Store the result in the gradient with respect to weight matrix
    tl.store(dw_ptr + o_idx[:, None] * I + i_idx[None, :], acc, mask=mask)

class TritonLinearFunction(torch.autograd.Function):
    """
    Autograd function for TritonLinear module to enable custom forward and backward passes.
    """
    
    @staticmethod
    def forward(ctx, x, weight):
        """
        Forward pass of the linear layer.

        Args:
            ctx: Context variable for backward pass.
            x (torch.Tensor): Input tensor of shape [B, I].
            weight (torch.Tensor): Weight tensor of shape [O, I].

        Returns:
            torch.Tensor: Output tensor of shape [B, O].

        Explanation:
            1. Get input dimensions from tensors.
            2. Allocate output tensor with appropriate shape.
            3. Launch Triton kernel for matrix multiplication.
            4. Save tensors and dimensions for backward pass.
            5. Return computed output.
        """
        # Get input dimensions
        B, I = x.shape
        O = weight.shape[0]
        
        # Allocate output tensor
        y = torch.empty((B, O), device=x.device, dtype=x.dtype)
        
        # Launch the kernel for forward pass
        grid = (triton.cdiv(B, 32), triton.cdiv(O, 32))
        linear_forward_kernel[grid](
            x.data_ptr(), weight.data_ptr(), y.data_ptr(),
            B, I, O,
        )
        
        # Save inputs for backward pass
        ctx.save_for_backward(x, weight)
        ctx.dims = (B, I, O)
        
        return y
    
    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass of the linear layer.

        Args:
            ctx: Context variable from forward pass.
            dy (torch.Tensor): Gradient of loss with respect to output of shape [B, O].

        Returns:
            tuple: (dx, dw) - Gradients with respect to input and weight.

        Explanation:
            1. Retrieve saved tensors and dimensions from context.
            2. Allocate tensors for input and weight gradients.
            3. Launch kernel to compute input gradients (dx = dy @ W).
            4. Launch kernel to compute weight gradients (dw = dy.T @ x).
            5. Return computed gradients.
        """
        # Get saved tensors and dimensions
        x, weight = ctx.saved_tensors
        B, I, O = ctx.dims
        
        # Allocate tensors for gradients
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        
        # Launch kernel for input gradients
        grid_dx = (triton.cdiv(B, 32), triton.cdiv(I, 32))
        linear_backward_input_kernel[grid_dx](
            dy.data_ptr(), weight.data_ptr(), dx.data_ptr(),
            B, I, O,
        )
        
        # Launch kernel for weight gradients
        grid_dw = (triton.cdiv(O, 32), triton.cdiv(I, 32))
        linear_backward_weight_kernel[grid_dw](
            x.data_ptr(), dy.data_ptr(), dw.data_ptr(),
            B, I, O,
        )
        
        return dx, dw

class TritonLinear(torch.nn.Module):
    """
    Linear layer implementation using Triton kernels.
    
    This module performs a linear transformation: y = x @ W.T
    """
    
    def __init__(self, in_features, out_features, bias=False, device='cuda', dtype=torch.float16):
        """
        Initialize the linear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool): If set to False, the layer will not learn an additive bias. Currently not supported.
            device (str): Device to create tensors on.
            dtype (torch.dtype): Data type of tensors.

        Returns:
            None

        Explanation:
            1. Initialize parent class.
            2. Store layer dimensions.
            3. Create weight parameter with appropriate shape.
            4. Initialize weights using Kaiming uniform.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Reset the parameters using Kaiming uniform initialization.

        Args:
            None

        Returns:
            None

        Explanation:
            1. Use PyTorch's Kaiming uniform initialization for weights.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        """
        Forward pass of the linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, in_features].

        Returns:
            torch.Tensor: Output tensor of shape [B, out_features].

        Explanation:
            1. Apply custom autograd function for forward and backward computation.
        """
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
