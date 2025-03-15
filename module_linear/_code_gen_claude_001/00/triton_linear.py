# filename: triton_linear.py
import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8}, num_stages=3),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}, num_stages=4),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def _forward_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, y_ptr,
    # Matrix dimensions
    B, I, O,
    # The stride variables represent how much we need to increase the ptr by to move by 1 element in a particular dimension
    stride_x_b, stride_x_i,
    stride_w_o, stride_w_i,
    stride_y_b, stride_y_o,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute Y = X @ W.T where X has shape (B, I) and W has shape (O, I).

    Args:
        x_ptr (pointer): Pointer to the input tensor X. [B, I]
        w_ptr (pointer): Pointer to the weight tensor W. [O, I]
        y_ptr (pointer): Pointer to the output tensor Y. [B, O]
        B (int): Batch size.
        I (int): Input feature dimension.
        O (int): Output feature dimension.
        stride_x_b (int): Stride of X along the batch dimension.
        stride_x_i (int): Stride of X along the input feature dimension.
        stride_w_o (int): Stride of W along the output feature dimension.
        stride_w_i (int): Stride of W along the input feature dimension.
        stride_y_b (int): Stride of Y along the batch dimension.
        stride_y_o (int): Stride of Y along the output feature dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for the batch dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for the input feature dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for the output feature dimension.
        GROUP_SIZE_B (tl.constexpr): Group size for the batch dimension.

    Explanation:
        1. Calculate program IDs and starting offsets for the current block.
        2. Create offsets and masks for the B and O dimensions.
        3. Initialize an accumulator tensor with zeros.
        4. Loop through the input dimension in blocks of BLOCK_SIZE_I.
        5. Load input and weight blocks into shared memory.
        6. Perform matrix multiplication (X @ W.T) for the current blocks.
        7. Accumulate the results in the accumulator.
        8. Store the final accumulated result to the output tensor.
    """
    # Program ID
    pid_b = tl.program_id(0)  # batch dimension
    pid_o = tl.program_id(1)  # output dimension
    
    # Calculate the starting offsets for this program instance
    start_b = pid_b * BLOCK_SIZE_B
    start_o = pid_o * BLOCK_SIZE_O
    
    # Create offsets for the B and O dimensions
    offs_b = start_b + tl.arange(0, BLOCK_SIZE_B)
    offs_o = start_o + tl.arange(0, BLOCK_SIZE_O)
    
    # Create a mask for bounds checking
    mask_b = offs_b < B
    mask_o = offs_o < O
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    # Iterate over the input dimension in blocks
    for i in range(0, I, BLOCK_SIZE_I):
        # Create offsets and mask for the I dimension
        offs_i = i + tl.arange(0, BLOCK_SIZE_I)
        mask_i = offs_i < I
        
        # Load blocks from X (input) and W (weights)
        x_block = tl.load(
            x_ptr + offs_b[:, None] * stride_x_b + offs_i[None, :] * stride_x_i,
            mask=mask_b[:, None] & mask_i[None, :], 
            other=0.0
        )
        w_block = tl.load(
            w_ptr + offs_o[:, None] * stride_w_o + offs_i[None, :] * stride_w_i,
            mask=mask_o[:, None] & mask_i[None, :], 
            other=0.0
        )
        
        # Compute matrix multiplication for this block
        # We need to take the transpose of W, so we'll compute x_block @ w_block.T
        acc += tl.dot(x_block, tl.trans(w_block))
    
    # Store the result
    mask = mask_b[:, None] & mask_o[None, :]
    tl.store(
        y_ptr + offs_b[:, None] * stride_y_b + offs_o[None, :] * stride_y_o,
        acc,
        mask=mask
    )

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8}, num_stages=3),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8}, num_stages=4),
    ],
    key=['B', 'O', 'I'],
)
@triton.jit
def _backward_input_kernel(
    # Pointers to matrices
    do_ptr, w_ptr, di_ptr,
    # Matrix dimensions
    B, O, I,
    # The stride variables represent how much we need to increase the ptr by to move by 1 element in a particular dimension
    stride_do_b, stride_do_o,
    stride_w_o, stride_w_i,
    stride_di_b, stride_di_i,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute d_input = d_output @ weight where d_output has shape (B, O) and weight has shape (O, I).

    Args:
        do_ptr (pointer): Pointer to the gradient tensor for output. [B, O]
        w_ptr (pointer): Pointer to the weight tensor W. [O, I]
        di_ptr (pointer): Pointer to the gradient tensor for input. [B, I]
        B (int): Batch size.
        O (int): Output feature dimension.
        I (int): Input feature dimension.
        stride_do_b (int): Stride of dY along the batch dimension.
        stride_do_o (int): Stride of dY along the output feature dimension.
        stride_w_o (int): Stride of W along the output feature dimension.
        stride_w_i (int): Stride of W along the input feature dimension.
        stride_di_b (int): Stride of dX along the batch dimension.
        stride_di_i (int): Stride of dX along the input feature dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for the batch dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for the output feature dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for the input feature dimension.
        GROUP_SIZE_B (tl.constexpr): Group size for the batch dimension.

    Explanation:
        1. Calculate program IDs and starting offsets for the current block.
        2. Create offsets and masks for the B and I dimensions.
        3. Initialize an accumulator tensor with zeros.
        4. Loop through the output dimension in blocks of BLOCK_SIZE_O.
        5. Load d_output and weight blocks into shared memory.
        6. Perform matrix multiplication (dY @ W) for the current blocks.
        7. Accumulate the results in the accumulator.
        8. Store the final accumulated result to the d_input tensor.
    """
    # Program ID
    pid_b = tl.program_id(0)  # batch dimension
    pid_i = tl.program_id(1)  # input dimension
    
    # Calculate the starting offsets for this program instance
    start_b = pid_b * BLOCK_SIZE_B
    start_i = pid_i * BLOCK_SIZE_I
    
    # Create offsets for the B and I dimensions
    offs_b = start_b + tl.arange(0, BLOCK_SIZE_B)
    offs_i = start_i + tl.arange(0, BLOCK_SIZE_I)
    
    # Create a mask for bounds checking
    mask_b = offs_b < B
    mask_i = offs_i < I
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over the output dimension in blocks
    for o in range(0, O, BLOCK_SIZE_O):
        # Create offsets and mask for the O dimension
        offs_o = o + tl.arange(0, BLOCK_SIZE_O)
        mask_o = offs_o < O
        
        # Load blocks from d_output and weight
        do_block = tl.load(
            do_ptr + offs_b[:, None] * stride_do_b + offs_o[None, :] * stride_do_o,
            mask=mask_b[:, None] & mask_o[None, :], 
            other=0.0
        )
        w_block = tl.load(
            w_ptr + offs_o[:, None] * stride_w_o + offs_i[None, :] * stride_w_i,
            mask=mask_o[:, None] & mask_i[None, :], 
            other=0.0
        )
        
        # Compute matrix multiplication for this block
        acc += tl.dot(do_block, w_block)
    
    # Store the result
    mask = mask_b[:, None] & mask_i[None, :]
    tl.store(
        di_ptr + offs_b[:, None] * stride_di_b + offs_i[None, :] * stride_di_i,
        acc,
        mask=mask
    )

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8}, num_stages=3),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8}, num_stages=4),
    ],
    key=['O', 'I', 'B'],
)
@triton.jit
def _backward_weight_kernel(
    # Pointers to matrices
    do_ptr, x_ptr, dw_ptr,
    # Matrix dimensions
    B, O, I,
    # The stride variables represent how much we need to increase the ptr by to move by 1 element in a particular dimension
    stride_do_b, stride_do_o,
    stride_x_b, stride_x_i,
    stride_dw_o, stride_dw_i,
    # Meta-parameters
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute d_weight = d_output.T @ input where d_output has shape (B, O) and input has shape (B, I).

    Args:
        do_ptr (pointer): Pointer to the gradient tensor for output. [B, O]
        x_ptr (pointer): Pointer to the input tensor X. [B, I]
        dw_ptr (pointer): Pointer to the gradient tensor for weights. [O, I]
        B (int): Batch size.
        O (int): Output feature dimension.
        I (int): Input feature dimension.
        stride_do_b (int): Stride of dY along the batch dimension.
        stride_do_o (int): Stride of dY along the output feature dimension.
        stride_x_b (int): Stride of X along the batch dimension.
        stride_x_i (int): Stride of X along the input feature dimension.
        stride_dw_o (int): Stride of dW along the output feature dimension.
        stride_dw_i (int): Stride of dW along the input feature dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for the output feature dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for the input feature dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for the batch dimension.
        GROUP_SIZE_O (tl.constexpr): Group size for the output dimension.

    Explanation:
        1. Calculate program IDs and starting offsets for the current block.
        2. Create offsets and masks for the O and I dimensions.
        3. Initialize an accumulator tensor with zeros.
        4. Loop through the batch dimension in blocks of BLOCK_SIZE_B.
        5. Load d_output and input blocks into shared memory.
        6. Perform matrix multiplication (dY.T @ X) for the current blocks.
        7. Accumulate the results in the accumulator.
        8. Store the final accumulated result to the d_weight tensor.
    """
    # Program ID
    pid_o = tl.program_id(0)  # output dimension
    pid_i = tl.program_id(1)  # input dimension
    
    # Calculate the starting offsets for this program instance
    start_o = pid_o * BLOCK_SIZE_O
    start_i = pid_i * BLOCK_SIZE_I
    
    # Create offsets for the O and I dimensions
    offs_o = start_o + tl.arange(0, BLOCK_SIZE_O)
    offs_i = start_i + tl.arange(0, BLOCK_SIZE_I)
    
    # Create a mask for bounds checking
    mask_o = offs_o < O
    mask_i = offs_i < I
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over the batch dimension in blocks
    for b in range(0, B, BLOCK_SIZE_B):
        # Create offsets and mask for the B dimension
        offs_b = b + tl.arange(0, BLOCK_SIZE_B)
        mask_b = offs_b < B
        
        # Load blocks from d_output and input
        do_block = tl.load(
            do_ptr + offs_b[:, None] * stride_do_b + offs_o[None, :] * stride_do_o,
            mask=mask_b[:, None] & mask_o[None, :], 
            other=0.0
        )
        x_block = tl.load(
            x_ptr + offs_b[:, None] * stride_x_b + offs_i[None, :] * stride_x_i,
            mask=mask_b[:, None] & mask_i[None, :], 
            other=0.0
        )
        
        # Compute matrix multiplication for this block
        # We need to compute do_block.T @ x_block
        acc += tl.dot(tl.trans(do_block), x_block)
    
    # Store the result
    mask = mask_o[:, None] & mask_i[None, :]
    tl.store(
        dw_ptr + offs_o[:, None] * stride_dw_o + offs_i[None, :] * stride_dw_i,
        acc,
        mask=mask
    )


def triton_linear_forward(x, weight):
    """
    Compute the forward pass of a linear layer: y = x @ weight.T
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, I). [B, I]
        weight (torch.Tensor): Weight tensor of shape (O, I). [O, I]
        
    Returns:
        torch.Tensor: Output tensor of shape (B, O). [B, O]
        
    Explanation:
        1. Extract shapes from input tensors.
        2. Verify that the input dimensions match.
        3. Allocate memory for the output tensor.
        4. Compute grid dimensions for the kernel launch.
        5. Launch the Triton kernel to perform the forward pass.
        6. Return the computed output tensor.
    """
    B, I = x.shape
    O, I_weight = weight.shape
    
    # Make sure the input dimensions match
    assert I == I_weight, f"Input dimension mismatch: {I} vs {I_weight}"
    
    # Allocate output
    y = torch.empty((B, O), device=x.device, dtype=x.dtype)
    
    # Launch the triton kernel for the forward pass
    grid = lambda meta: (
        triton.cdiv(B, meta['BLOCK_SIZE_B']),  # blocks in the B dimension
        triton.cdiv(O, meta['BLOCK_SIZE_O']),  # blocks in the O dimension
    )
    
    _forward_kernel[grid](
        x, weight, y,
        B, I, O,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
    )
    
    return y


def triton_linear_backward(grad_output, x, weight):
    """
    Compute the backward pass of a linear layer.
    
    Args:
        grad_output (torch.Tensor): Gradient of the output tensor of shape (B, O). [B, O]
        x (torch.Tensor): Input tensor of shape (B, I). [B, I]
        weight (torch.Tensor): Weight tensor of shape (O, I). [O, I]
        
    Returns:
        tuple: A tuple containing:
            - grad_input (torch.Tensor): Gradient of the input tensor of shape (B, I). [B, I]
            - grad_weight (torch.Tensor): Gradient of the weight tensor of shape (O, I). [O, I]
            
    Explanation:
        1. Extract shapes from input tensors.
        2. Verify that the dimensions match across all tensors.
        3. Allocate memory for gradient tensors.
        4. Compute grid dimensions for the input gradient kernel.
        5. Launch the kernel to compute gradients with respect to input.
        6. Compute grid dimensions for the weight gradient kernel.
        7. Launch the kernel to compute gradients with respect to weights.
        8. Return both computed gradient tensors.
    """
    B, O = grad_output.shape
    B_x, I = x.shape
    O_w, I_w = weight.shape
    
    # Make sure the dimensions match
    assert B == B_x, f"Batch dimension mismatch: {B} vs {B_x}"
    assert O == O_w, f"Output dimension mismatch: {O} vs {O_w}"
    assert I == I_w, f"Input dimension mismatch: {I} vs {I_w}"
    
    # Allocate gradients
    grad_input = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)
    
    # Launch the triton kernel for the backward pass with respect to the input
    grid_input = lambda meta: (
        triton.cdiv(B, meta['BLOCK_SIZE_B']),  # blocks in the B dimension
        triton.cdiv(I, meta['BLOCK_SIZE_I']),  # blocks in the I dimension
    )
    
    _backward_input_kernel[grid_input](
        grad_output, weight, grad_input,
        B, O, I,
        grad_output.stride(0), grad_output.stride(1),
        weight.stride(0), weight.stride(1),
        grad_input.stride(0), grad_input.stride(1),
    )
    
    # Launch the triton kernel for the backward pass with respect to the weight
    grid_weight = lambda meta: (
        triton.cdiv(O, meta['BLOCK_SIZE_O']),  # blocks in the O dimension
        triton.cdiv(I, meta['BLOCK_SIZE_I']),  # blocks in the I dimension
    )
    
    _backward_weight_kernel[grid_weight](
        grad_output, x, grad_weight,
        B, O, I,
        grad_output.stride(0), grad_output.stride(1),
        x.stride(0), x.stride(1),
        grad_weight.stride(0), grad_weight.stride(1),
    )
    
    return grad_input, grad_weight


class TritonLinearFunction(torch.autograd.Function):
    """
    Autograd function for the TritonLinear module.
    
    Explanation:
        1. In forward, save input and weight tensors for backward pass.
        2. Call the triton forward kernel to compute the output.
        3. In backward, retrieve saved tensors and compute gradients.
        4. Return gradients for all inputs that require gradients.
    """
    @staticmethod
    def forward(ctx, x, weight):
        """
        Forward pass of the linear layer.
        
        Args:
            ctx (torch.autograd.function.Context): Context to save tensors for backward.
            x (torch.Tensor): Input tensor of shape (B, I). [B, I]
            weight (torch.Tensor): Weight tensor of shape (O, I). [O, I]
            
        Returns:
            torch.Tensor: Output tensor of shape (B, O). [B, O]
        """
        ctx.save_for_backward(x, weight)
        return triton_linear_forward(x, weight)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the linear layer.
        
        Args:
            ctx (torch.autograd.function.Context): Context containing saved tensors.
            grad_output (torch.Tensor): Gradient of the output tensor. [B, O]
            
        Returns:
            tuple: Gradients for each input:
                - grad_input (torch.Tensor): Gradient w.r.t. input. [B, I]
                - grad_weight (torch.Tensor): Gradient w.r.t. weight. [O, I]
        """
        x, weight = ctx.saved_tensors
        grad_input, grad_weight = triton_linear_backward(grad_output, x, weight)
        return grad_input, grad_weight


class TritonLinear(torch.nn.Module):
    """
    A linear module implementation using Triton kernels.
    
    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        weight (torch.nn.Parameter): Learnable weights of the module of shape (out_features, in_features).
        
    Explanation:
        1. Initializes a linear layer with specified input and output dimensions.
        2. Creates a weight parameter with appropriate shape.
        3. Initializes the weight parameter using kaiming_uniform.
        4. During forward pass, applies the TritonLinearFunction to input and weight.
    """
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        """
        Initialize the TritonLinear module.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: False.
            device (torch.device, optional): Device to place the parameters. Default: None.
            dtype (torch.dtype, optional): Data type of the parameters. Default: None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize the weight parameter
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()
        
        if bias:
            raise NotImplementedError("Bias is not supported for TritonLinear yet.")
    
    def reset_parameters(self):
        """
        Reset the parameters of the linear layer.
        
        Explanation:
            1. Initializes the weight parameter using kaiming_uniform.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        """
        Forward pass of the linear layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, I). [B, I]
            
        Returns:
            torch.Tensor: Output tensor of shape (B, O). [B, O]
        """
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
