# filename: linear_triton.py
import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_O': 16, 'GROUP_SIZE_B': 1}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 2}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_forward_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    # Matrix dimensions
    B, I, O,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    """
    Compute the forward pass for a linear layer: output = input @ weight.T + bias
    
    Args:
        input_ptr (pointer): Pointer to the input matrix of shape (B, I). [B×I]
        weight_ptr (pointer): Pointer to the weight matrix of shape (O, I). [O×I]
        output_ptr (pointer): Pointer to the output matrix of shape (B, O). [B×O]
        bias_ptr (pointer): Pointer to the bias vector of shape (O,) or None. [O]
        B (int): Batch size
        I (int): Input features
        O (int): Output features
        stride_input_b (int): Stride for moving along the B dimension of input
        stride_input_i (int): Stride for moving along the I dimension of input
        stride_weight_o (int): Stride for moving along the O dimension of weight
        stride_weight_i (int): Stride for moving along the I dimension of weight
        stride_output_b (int): Stride for moving along the B dimension of output
        stride_output_o (int): Stride for moving along the O dimension of output
        BLOCK_SIZE_B (int): Block size for batch dimension
        BLOCK_SIZE_I (int): Block size for input feature dimension
        BLOCK_SIZE_O (int): Block size for output feature dimension
        GROUP_SIZE_B (int): Group size for batch dimension
        USE_BIAS (bool): Whether to use bias
    
    Explanation:
        1. Compute the program IDs and determine which section of the output matrix to compute
        2. Create array indices and masks for batch and output dimensions
        3. Load bias values if specified
        4. Initialize accumulator for matrix multiplication
        5. Iterate over blocks of the input dimension, loading relevant slices of input and weight
        6. Compute partial matrix multiplication and accumulate results
        7. Convert accumulator to output data type
        8. Add bias if needed
        9. Store results to output matrix
    """
    # -----------------------------------------------------------
    # Program ID
    # Batch dimension: program ID for B
    pid_b = tl.program_id(axis=0)
    # Group size: multiple program_id's handle different parts of the output
    pid_batch_group = pid_b // GROUP_SIZE_B
    pid_in_group = pid_b % GROUP_SIZE_B
    # Output dimension: program ID for O
    pid_o = tl.program_id(axis=1)
    
    # -----------------------------------------------------------
    # Compute batch and output offsets for this program
    # Batch offset
    batch_offset_base = pid_batch_group * BLOCK_SIZE_B
    # Output offset
    output_offset_base = pid_o * BLOCK_SIZE_O
    
    # -----------------------------------------------------------
    # Batch indices
    batch_indices = batch_offset_base + tl.arange(0, BLOCK_SIZE_B)
    # Only consider valid batch indices
    batch_mask = batch_indices < B
    # Filter out invalid batch indices
    batch_indices = tl.where(batch_mask, batch_indices, 0)
    
    # Output feature indices
    output_indices = output_offset_base + tl.arange(0, BLOCK_SIZE_O)
    # Only consider valid output feature indices
    output_mask = output_indices < O
    # Filter out invalid output feature indices
    output_indices = tl.where(output_mask, output_indices, 0)
    
    # -----------------------------------------------------------
    # Load bias if needed
    if USE_BIAS:
        bias = tl.load(bias_ptr + output_indices, mask=output_mask)
    
    # -----------------------------------------------------------
    # Compute linear
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    # We iterate over blocks of the input feature dimension
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        # Input feature indices
        input_offset_base = i * BLOCK_SIZE_I
        input_indices = input_offset_base + tl.arange(0, BLOCK_SIZE_I)
        # Load input input/features
        input_mask = input_indices < I
        input_indices = tl.where(input_mask, input_indices, 0)
        
        # Pointers to input and weight for this block
        input_ptrs = input_ptr + (batch_indices[:, None] * stride_input_b) + (input_indices[None, :] * stride_input_i)
        weight_ptrs = weight_ptr + (output_indices[:, None] * stride_weight_o) + (input_indices[None, :] * stride_weight_i)
        
        # Load inputs and weights
        x = tl.load(input_ptrs, mask=batch_mask[:, None] & input_mask[None, :])
        w = tl.load(weight_ptrs, mask=output_mask[:, None] & input_mask[None, :])
        
        # Compute
        acc += tl.dot(x, tl.trans(w))
    
    # Convert accumulator to the right data type
    acc = acc.to(tl.get_dtype(input_ptr))
    
    # Add bias if needed
    if USE_BIAS:
        acc += bias[None, :]
    
    # Write output
    output_ptrs = output_ptr + (batch_indices[:, None] * stride_output_b) + (output_indices[None, :] * stride_output_o)
    tl.store(output_ptrs, acc, mask=batch_mask[:, None] & output_mask[None, :])


def linear_forward(x, weight, bias=None):
    """
    Forward pass for Triton Linear implementation
    
    Args:
        x (torch.Tensor): Input tensor of shape [B, I]. [B×I]
        weight (torch.Tensor): Weight matrix of shape [O, I]. [O×I]
        bias (torch.Tensor, optional): Bias vector of shape [O]. [O]
    
    Returns:
        torch.Tensor: Output tensor of shape [B, O]. [B×O]
    
    Explanation:
        1. Extract dimensions from input and weight tensors
        2. Validate that input feature dimensions match
        3. Create output tensor with appropriate shape
        4. Determine if bias should be used
        5. Set up grid configuration for Triton kernel
        6. Launch the kernel with appropriate parameters
        7. Return the computed output
    """
    # Input is (B, I), weight is (O, I)
    batch_size, in_features = x.shape
    out_features, in_features_w = weight.shape
    
    assert in_features == in_features_w, "Input feature dimensions don't match"
    
    # Create output tensor
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    # Set flag for bias
    use_bias = bias is not None
    
    # Default bias pointer (will be ignored if use_bias is False)
    bias_ptr = 0
    if use_bias:
        bias_ptr = bias.data_ptr()
    
    # Launch the kernel
    grid = lambda META: (
        triton.cdiv(batch_size, META['BLOCK_SIZE_B']) * META['GROUP_SIZE_B'],
        triton.cdiv(out_features, META['BLOCK_SIZE_O']),
    )
    linear_forward_kernel[grid](
        x.data_ptr(), weight.data_ptr(), output.data_ptr(), bias_ptr,
        batch_size, in_features, out_features,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        USE_BIAS=use_bias,
    )
    
    return output


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_O': 16, 'GROUP_SIZE_B': 1}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 2}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_dx_kernel(
    # Pointers to matrices
    grad_output_ptr, weight_ptr, grad_input_ptr,
    # Matrix dimensions
    B, O, I,
    # The stride variables
    stride_grad_output_b, stride_grad_output_o,
    stride_weight_o, stride_weight_i,
    stride_grad_input_b, stride_grad_input_i,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the backward pass for the input gradients: dx = dy @ w
    
    Args:
        grad_output_ptr (pointer): Pointer to the gradient w.r.t. output of shape (B, O). [B×O]
        weight_ptr (pointer): Pointer to the weight matrix of shape (O, I). [O×I]
        grad_input_ptr (pointer): Pointer to the gradient w.r.t. input of shape (B, I). [B×I]
        B (int): Batch size
        O (int): Output features
        I (int): Input features
        stride_grad_output_b (int): Stride for moving along the B dimension of grad_output
        stride_grad_output_o (int): Stride for moving along the O dimension of grad_output
        stride_weight_o (int): Stride for moving along the O dimension of weight
        stride_weight_i (int): Stride for moving along the I dimension of weight
        stride_grad_input_b (int): Stride for moving along the B dimension of grad_input
        stride_grad_input_i (int): Stride for moving along the I dimension of grad_input
        BLOCK_SIZE_B (int): Block size for batch dimension
        BLOCK_SIZE_I (int): Block size for input feature dimension
        BLOCK_SIZE_O (int): Block size for output feature dimension
        GROUP_SIZE_B (int): Group size for batch dimension
    
    Explanation:
        1. Compute the program IDs and determine which section of the gradient matrix to compute
        2. Create array indices and masks for batch and input dimensions
        3. Initialize accumulator for gradient computation
        4. Iterate over blocks of the output dimension, loading relevant slices of grad_output and weight
        5. Compute partial gradient contribution and accumulate results
        6. Convert accumulator to proper data type
        7. Store results to grad_input matrix
    """
    # -----------------------------------------------------------
    # Program ID
    # Batch dimension: program ID for B
    pid_b = tl.program_id(axis=0)
    # Group size: multiple program_id's handle different parts of the output
    pid_batch_group = pid_b // GROUP_SIZE_B
    pid_in_group = pid_b % GROUP_SIZE_B
    # Input dimension: program ID for I
    pid_i = tl.program_id(axis=1)
    
    # -----------------------------------------------------------
    # Compute batch and input offsets for this program
    # Batch offset
    batch_offset_base = pid_batch_group * BLOCK_SIZE_B
    # Input offset
    input_offset_base = pid_i * BLOCK_SIZE_I
    
    # -----------------------------------------------------------
    # Batch indices
    batch_indices = batch_offset_base + tl.arange(0, BLOCK_SIZE_B)
    # Only consider valid batch indices
    batch_mask = batch_indices < B
    # Filter out invalid batch indices
    batch_indices = tl.where(batch_mask, batch_indices, 0)
    
    # Input feature indices
    input_indices = input_offset_base + tl.arange(0, BLOCK_SIZE_I)
    # Only consider valid input feature indices
    input_mask = input_indices < I
    # Filter out invalid input feature indices
    input_indices = tl.where(input_mask, input_indices, 0)
    
    # -----------------------------------------------------------
    # Compute grad_input = grad_output @ weight
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    # We iterate over blocks of the output feature dimension
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        # Output feature indices
        output_offset_base = o * BLOCK_SIZE_O
        output_indices = output_offset_base + tl.arange(0, BLOCK_SIZE_O)
        # Create mask for valid output indices
        output_mask = output_indices < O
        output_indices = tl.where(output_mask, output_indices, 0)
        
        # Pointers to grad_output and weight for this block
        grad_output_ptrs = grad_output_ptr + (batch_indices[:, None] * stride_grad_output_b) + (output_indices[None, :] * stride_grad_output_o)
        weight_ptrs = weight_ptr + (output_indices[:, None] * stride_weight_o) + (input_indices[None, :] * stride_weight_i)
        
        # Load grad_output and weights
        grad_output = tl.load(grad_output_ptrs, mask=batch_mask[:, None] & output_mask[None, :])
        weight = tl.load(weight_ptrs, mask=output_mask[:, None] & input_mask[None, :])
        
        # Compute
        acc += tl.dot(grad_output, weight)
    
    # Convert accumulator to the right data type
    acc = acc.to(tl.get_dtype(grad_input_ptr))
    
    # Write output
    grad_input_ptrs = grad_input_ptr + (batch_indices[:, None] * stride_grad_input_b) + (input_indices[None, :] * stride_grad_input_i)
    tl.store(grad_input_ptrs, acc, mask=batch_mask[:, None] & input_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_B': 16, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 2}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 4}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_dw_kernel(
    # Pointers to matrices
    input_ptr, grad_output_ptr, grad_weight_ptr,
    # Matrix dimensions
    B, I, O,
    # The stride variables
    stride_input_b, stride_input_i,
    stride_grad_output_b, stride_grad_output_o,
    stride_grad_weight_o, stride_grad_weight_i,
    # Meta-parameters
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute the backward pass for the weight gradients: dw = dy.T @ x
    
    Args:
        input_ptr (pointer): Pointer to the input matrix of shape (B, I). [B×I]
        grad_output_ptr (pointer): Pointer to the gradient w.r.t. output of shape (B, O). [B×O]
        grad_weight_ptr (pointer): Pointer to the gradient w.r.t. weight of shape (O, I). [O×I]
        B (int): Batch size
        I (int): Input features
        O (int): Output features
        stride_input_b (int): Stride for moving along the B dimension of input
        stride_input_i (int): Stride for moving along the I dimension of input
        stride_grad_output_b (int): Stride for moving along the B dimension of grad_output
        stride_grad_output_o (int): Stride for moving along the O dimension of grad_output
        stride_grad_weight_o (int): Stride for moving along the O dimension of grad_weight
        stride_grad_weight_i (int): Stride for moving along the I dimension of grad_weight
        BLOCK_SIZE_O (int): Block size for output feature dimension
        BLOCK_SIZE_I (int): Block size for input feature dimension
        BLOCK_SIZE_B (int): Block size for batch dimension
        GROUP_SIZE_O (int): Group size for output dimension
    
    Explanation:
        1. Compute the program IDs and determine which section of the weight gradient to compute
        2. Create array indices and masks for output and input dimensions
        3. Initialize accumulator for weight gradient computation
        4. Iterate over blocks of the batch dimension, loading relevant slices of input and grad_output
        5. Compute partial weight gradient and accumulate results
        6. Convert accumulator to proper data type
        7. Store results to grad_weight matrix
    """
    # -----------------------------------------------------------
    # Program ID
    # Output dimension: program ID for O
    pid_o = tl.program_id(axis=0)
    # Group size: multiple program_id's handle different parts of the output
    pid_output_group = pid_o // GROUP_SIZE_O
    pid_in_group = pid_o % GROUP_SIZE_O
    # Input dimension: program ID for I
    pid_i = tl.program_id(axis=1)
    
    # -----------------------------------------------------------
    # Compute output and input offsets for this program
    # Output offset
    output_offset_base = pid_output_group * BLOCK_SIZE_O
    # Input offset
    input_offset_base = pid_i * BLOCK_SIZE_I
    
    # -----------------------------------------------------------
    # Output feature indices
    output_indices = output_offset_base + tl.arange(0, BLOCK_SIZE_O)
    # Only consider valid output feature indices
    output_mask = output_indices < O
    # Filter out invalid output feature indices
    output_indices = tl.where(output_mask, output_indices, 0)
    
    # Input feature indices
    input_indices = input_offset_base + tl.arange(0, BLOCK_SIZE_I)
    # Only consider valid input feature indices
    input_mask = input_indices < I
    # Filter out invalid input feature indices
    input_indices = tl.where(input_mask, input_indices, 0)
    
    # -----------------------------------------------------------
    # Compute grad_weight = grad_output.T @ input
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    # We iterate over blocks of the batch dimension
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        # Batch indices
        batch_offset_base = b * BLOCK_SIZE_B
        batch_indices = batch_offset_base + tl.arange(0, BLOCK_SIZE_B)
        # Create mask for valid batch indices
        batch_mask = batch_indices < B
        batch_indices = tl.where(batch_mask, batch_indices, 0)
        
        # Pointers to input and grad_output for this block
        input_ptrs = input_ptr + (batch_indices[:, None] * stride_input_b) + (input_indices[None, :] * stride_input_i)
        grad_output_ptrs = grad_output_ptr + (batch_indices[:, None] * stride_grad_output_b) + (output_indices[None, :] * stride_grad_output_o)
        
        # Load input and grad_output
        x = tl.load(input_ptrs, mask=batch_mask[:, None] & input_mask[None, :])
        grad_output = tl.load(grad_output_ptrs, mask=batch_mask[:, None] & output_mask[None, :])
        
        # Compute
        acc += tl.dot(tl.trans(grad_output), x)
    
    # Convert accumulator to the right data type
    acc = acc.to(tl.get_dtype(grad_weight_ptr))
    
    # Write output
    grad_weight_ptrs = grad_weight_ptr + (output_indices[:, None] * stride_grad_weight_o) + (input_indices[None, :] * stride_grad_weight_i)
    tl.store(grad_weight_ptrs, acc, mask=output_mask[:, None] & input_mask[None, :])


class LinearFunction(torch.autograd.Function):
    """
    Autograd function for linear transformation using Triton kernels
    """
    
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        """
        Forward pass for linear layer using Triton kernels.
        
        Args:
            ctx: Context object to save tensors for backward
            input (torch.Tensor): Input tensor of shape [B, I]. [B×I]
            weight (torch.Tensor): Weight matrix of shape [O, I]. [O×I]
            bias (torch.Tensor, optional): Bias vector of shape [O]. [O]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, O]. [B×O]
        
        Explanation:
            1. Save input tensors for use in backward pass
            2. Compute forward pass using Triton kernel
            3. Return computed output
        """
        # Save tensors for backward
        ctx.save_for_backward(input, weight, bias)
        # Compute forward
        output = linear_forward(input, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for linear layer using Triton kernels.
        
        Args:
            ctx: Context object with saved tensors
            grad_output (torch.Tensor): Gradient w.r.t. output tensor of shape [B, O]. [B×O]
            
        Returns:
            tuple: Tuple containing gradients for input, weight, and bias. [B×I], [O×I], [O]
        
        Explanation:
            1. Retrieve saved tensors from forward pass
            2. Extract dimensions from tensors
            3. Create empty tensors for input and weight gradients
            4. Compute input gradient (dx) using Triton kernel
            5. Compute weight gradient (dw) using Triton kernel
            6. Compute bias gradient (db) if bias was used in forward pass
            7. Return all gradients
        """
        # Retrieve saved tensors
        input, weight, bias = ctx.saved_tensors
        batch_size, in_features = input.shape
        out_features = weight.shape[0]
        
        # Compute input gradient: dx = dy @ w
        grad_input = torch.empty_like(input)
        grid_dx = lambda META: (
            triton.cdiv(batch_size, META['BLOCK_SIZE_B']) * META['GROUP_SIZE_B'],
            triton.cdiv(in_features, META['BLOCK_SIZE_I']),
        )
        linear_backward_dx_kernel[grid_dx](
            grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr(),
            batch_size, out_features, in_features,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1),
        )
        
        # Compute weight gradient: dw = dy.T @ x
        grad_weight = torch.empty_like(weight)
        grid_dw = lambda META: (
            triton.cdiv(out_features, META['BLOCK_SIZE_O']) * META['GROUP_SIZE_O'],
            triton.cdiv(in_features, META['BLOCK_SIZE_I']),
        )
        linear_backward_dw_kernel[grid_dw](
            input.data_ptr(), grad_output.data_ptr(), grad_weight.data_ptr(),
            batch_size, in_features, out_features,
            input.stride(0), input.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
        )
        
        # Compute bias gradient: db = sum(dy, dim=0)
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias


class TritonLinear(torch.nn.Module):
    """
    Linear layer implementation using Triton kernels for forward and backward pass.
    """
    
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        """
        Initializes a linear layer with Triton kernels.
        
        Args:
            in_features (int): Size of each input sample
            out_features (int): Size of each output sample
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: False
            device (torch.device, optional): Device to use
            dtype (torch.dtype, optional): Data type to use
        
        Explanation:
            1. Initialize parent class
            2. Store input and output dimension sizes
            3. Create weight parameter with appropriate shape
            4. Create bias parameter if requested
            5. Initialize parameters with values similar to PyTorch's implementation
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight parameter
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), 
                                                   device=device, dtype=dtype))
        
        # Create bias parameter if needed
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, 
                                                     device=device, dtype=dtype))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initializes parameters using the same method as PyTorch's Linear layer.
        
        Explanation:
            1. Initialize weights using Kaiming uniform initialization
            2. If bias is present, initialize it with values from a uniform distribution
            3. The bound for the uniform distribution is calculated based on input dimension
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        Forward pass of the linear layer using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, I]. [B×I]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, O]. [B×O]
        
        Explanation:
            1. Call the custom LinearFunction's apply method
            2. Pass input, weight, and bias to the function
            3. Return the computed output
        """
        return LinearFunction.apply(x, self.weight, self.bias)


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
