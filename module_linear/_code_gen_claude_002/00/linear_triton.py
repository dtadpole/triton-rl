# filename: linear_triton.py
import torch
import triton
import triton.language as tl
from torch.autograd import Function
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_O': 16, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_O': 16, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}),
        # Asymmetric configurations with super blocks
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 4, 'GROUP_SIZE_O': 4}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 4}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_I': 4, 'GROUP_SIZE_O': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute the forward pass of a linear layer: output = input @ weight.T
    
    Args:
        input_ptr (ptr): Pointer to the input tensor. [shape: (B, I)]
        weight_ptr (ptr): Pointer to the weight tensor. [shape: (O, I)]
        output_ptr (ptr): Pointer to the output tensor. [shape: (B, O)]
        B (int): Batch dimension size
        I (int): Input dimension size
        O (int): Output dimension size
        stride_input_b (int): Stride for moving in batch dimension of input tensor
        stride_input_i (int): Stride for moving in input dimension of input tensor
        stride_weight_o (int): Stride for moving in output dimension of weight tensor
        stride_weight_i (int): Stride for moving in input dimension of weight tensor
        stride_output_b (int): Stride for moving in batch dimension of output tensor
        stride_output_o (int): Stride for moving in output dimension of output tensor
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension in super blocks
        GROUP_SIZE_I (tl.constexpr): Group size for input dimension in super blocks
        GROUP_SIZE_O (tl.constexpr): Group size for output dimension in super blocks
    
    Explanation:
        1. Compute program IDs and offsets for the current block.
        2. Create pointers and masks for the batch and output dimensions.
        3. Initialize accumulator with zeros.
        4. Iterate through blocks of the input dimension.
        5. Load input and weight blocks with appropriate masking.
        6. Compute matrix multiplication for the current block.
        7. Store the accumulated results to the output tensor.
    """
    # Programs are grouped by batch and output dimension
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    # Compute superblock indices and offsets
    pid_b_sup = pid_b // GROUP_SIZE_B
    pid_o_sup = pid_o // GROUP_SIZE_O
    
    pid_b_sub = pid_b % GROUP_SIZE_B
    pid_o_sub = pid_o % GROUP_SIZE_O
    
    # Calculate memory offsets for superblocks and within superblocks
    b_offset_sup = pid_b_sup * BLOCK_SIZE_B * GROUP_SIZE_B
    o_offset_sup = pid_o_sup * BLOCK_SIZE_O * GROUP_SIZE_O
    
    b_offset_sub = pid_b_sub * BLOCK_SIZE_B
    o_offset_sub = pid_o_sub * BLOCK_SIZE_O
    
    # Total memory offsets
    b_offset = b_offset_sup + b_offset_sub
    o_offset = o_offset_sup + o_offset_sub
    
    # Create block pointers for the batch and output dimensions
    batch_range = b_offset + tl.arange(0, BLOCK_SIZE_B)
    out_range = o_offset + tl.arange(0, BLOCK_SIZE_O)
    
    # Mask out elements that are outside the bounds of the matrices
    batch_mask = batch_range < B
    out_mask = out_range < O
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    # Iterate through the input dimension in blocks
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        i_start = i * BLOCK_SIZE_I
        
        # Create pointers to the input and weight blocks
        input_ptrs = input_ptr + (batch_range[:, None] * stride_input_b + 
                                 (i_start + tl.arange(0, BLOCK_SIZE_I)[None, :]) * stride_input_i)
        weight_ptrs = weight_ptr + (out_range[:, None] * stride_weight_o + 
                                   (i_start + tl.arange(0, BLOCK_SIZE_I)[None, :]) * stride_weight_i)
        
        # Create masks for the input dimension
        i_mask = (i_start + tl.arange(0, BLOCK_SIZE_I)) < I
        
        # Load input and weight blocks with appropriate masking
        input_block = tl.load(input_ptrs, mask=batch_mask[:, None] & i_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptrs, mask=out_mask[:, None] & i_mask[None, :], other=0.0)
        
        # Compute matrix multiplication for this block: input @ weight.T
        acc += tl.dot(input_block, tl.trans(weight_block))
    
    # Store the results
    output_ptrs = output_ptr + batch_range[:, None] * stride_output_b + out_range[None, :] * stride_output_o
    tl.store(output_ptrs, acc, mask=batch_mask[:, None] & out_mask[None, :])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_O': 16, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_O': 16, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}),
        # Asymmetric configurations with super blocks
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 4, 'GROUP_SIZE_O': 4}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 4}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_I': 4, 'GROUP_SIZE_O': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def backward_input_kernel(
    grad_output_ptr, weight_ptr, grad_input_ptr,
    B, I, O,
    stride_grad_output_b, stride_grad_output_o,
    stride_weight_o, stride_weight_i,
    stride_grad_input_b, stride_grad_input_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute the gradients with respect to the input tensor: grad_input = grad_output @ weight
    
    Args:
        grad_output_ptr (ptr): Pointer to the gradient of the output tensor. [shape: (B, O)]
        weight_ptr (ptr): Pointer to the weight tensor. [shape: (O, I)]
        grad_input_ptr (ptr): Pointer to store the gradient of the input tensor. [shape: (B, I)]
        B (int): Batch dimension size
        I (int): Input dimension size
        O (int): Output dimension size
        stride_grad_output_b (int): Stride for moving in batch dimension of grad_output tensor
        stride_grad_output_o (int): Stride for moving in output dimension of grad_output tensor
        stride_weight_o (int): Stride for moving in output dimension of weight tensor
        stride_weight_i (int): Stride for moving in input dimension of weight tensor
        stride_grad_input_b (int): Stride for moving in batch dimension of grad_input tensor
        stride_grad_input_i (int): Stride for moving in input dimension of grad_input tensor
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension in super blocks
        GROUP_SIZE_I (tl.constexpr): Group size for input dimension in super blocks
        GROUP_SIZE_O (tl.constexpr): Group size for output dimension in super blocks
    
    Explanation:
        1. Compute program IDs and offsets for the current block.
        2. Create pointers and masks for the batch and input dimensions.
        3. Initialize accumulator with zeros.
        4. Iterate through blocks of the output dimension.
        5. Load grad_output and weight blocks with appropriate masking.
        6. Compute matrix multiplication: grad_input = grad_output @ weight.
        7. Store the accumulated results to the grad_input tensor.
    """
    # Programs are grouped by batch and input dimension
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    # Compute superblock indices and offsets
    pid_b_sup = pid_b // GROUP_SIZE_B
    pid_i_sup = pid_i // GROUP_SIZE_I
    
    pid_b_sub = pid_b % GROUP_SIZE_B
    pid_i_sub = pid_i % GROUP_SIZE_I
    
    # Calculate memory offsets for superblocks and within superblocks
    b_offset_sup = pid_b_sup * BLOCK_SIZE_B * GROUP_SIZE_B
    i_offset_sup = pid_i_sup * BLOCK_SIZE_I * GROUP_SIZE_I
    
    b_offset_sub = pid_b_sub * BLOCK_SIZE_B
    i_offset_sub = pid_i_sub * BLOCK_SIZE_I
    
    # Total memory offsets
    b_offset = b_offset_sup + b_offset_sub
    i_offset = i_offset_sup + i_offset_sub
    
    # Create block pointers for the batch and input dimensions
    batch_range = b_offset + tl.arange(0, BLOCK_SIZE_B)
    in_range = i_offset + tl.arange(0, BLOCK_SIZE_I)
    
    # Mask out elements that are outside the bounds
    batch_mask = batch_range < B
    in_mask = in_range < I
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate through output dimension in blocks
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        o_start = o * BLOCK_SIZE_O
        
        # Create pointers to grad_output and weight blocks
        grad_output_ptrs = grad_output_ptr + (batch_range[:, None] * stride_grad_output_b + 
                                             (o_start + tl.arange(0, BLOCK_SIZE_O)[None, :]) * stride_grad_output_o)
        weight_ptrs = weight_ptr + ((o_start + tl.arange(0, BLOCK_SIZE_O))[:, None] * stride_weight_o +
                                   in_range[None, :] * stride_weight_i)
        
        # Create masks for the output dimension
        o_mask = (o_start + tl.arange(0, BLOCK_SIZE_O)) < O
        
        # Load grad_output and weight blocks with appropriate masking
        grad_output_block = tl.load(grad_output_ptrs, mask=batch_mask[:, None] & o_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptrs, mask=o_mask[:, None] & in_mask[None, :], other=0.0)
        
        # Compute matrix multiplication: grad_input = grad_output @ weight
        acc += tl.dot(grad_output_block, weight_block)
    
    # Store the results
    grad_input_ptrs = grad_input_ptr + batch_range[:, None] * stride_grad_input_b + in_range[None, :] * stride_grad_input_i
    tl.store(grad_input_ptrs, acc, mask=batch_mask[:, None] & in_mask[None, :])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_B': 16, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_B': 1}),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_B': 1}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_B': 1}),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_B': 1}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_B': 1}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_B': 1}),
        triton.Config({'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_B': 16, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}),
        # Asymmetric configurations with super blocks
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 4, 'GROUP_SIZE_B': 4}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 4, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 4}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 4, 'GROUP_SIZE_I': 4, 'GROUP_SIZE_B': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def backward_weight_kernel(
    grad_output_ptr, input_ptr, grad_weight_ptr,
    B, I, O,
    stride_grad_output_b, stride_grad_output_o,
    stride_input_b, stride_input_i,
    stride_grad_weight_o, stride_grad_weight_i,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the gradients with respect to the weight tensor: grad_weight = grad_output.T @ input
    
    Args:
        grad_output_ptr (ptr): Pointer to the gradient of the output tensor. [shape: (B, O)]
        input_ptr (ptr): Pointer to the input tensor. [shape: (B, I)]
        grad_weight_ptr (ptr): Pointer to store the gradient of the weight tensor. [shape: (O, I)]
        B (int): Batch dimension size
        I (int): Input dimension size
        O (int): Output dimension size
        stride_grad_output_b (int): Stride for moving in batch dimension of grad_output tensor
        stride_grad_output_o (int): Stride for moving in output dimension of grad_output tensor
        stride_input_b (int): Stride for moving in batch dimension of input tensor
        stride_input_i (int): Stride for moving in input dimension of input tensor
        stride_grad_weight_o (int): Stride for moving in output dimension of grad_weight tensor
        stride_grad_weight_i (int): Stride for moving in input dimension of grad_weight tensor
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        GROUP_SIZE_O (tl.constexpr): Group size for output dimension in super blocks
        GROUP_SIZE_I (tl.constexpr): Group size for input dimension in super blocks
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension in super blocks
    
    Explanation:
        1. Compute program IDs and offsets for the current block.
        2. Create pointers and masks for the output and input dimensions.
        3. Initialize accumulator with zeros.
        4. Iterate through blocks of the batch dimension.
        5. Load grad_output and input blocks with appropriate masking.
        6. Compute matrix multiplication: grad_weight = grad_output.T @ input.
        7. Store the accumulated results to the grad_weight tensor.
    """
    # Programs are grouped by output and input dimension
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    # Compute superblock indices and offsets
    pid_o_sup = pid_o // GROUP_SIZE_O
    pid_i_sup = pid_i // GROUP_SIZE_I
    
    pid_o_sub = pid_o % GROUP_SIZE_O
    pid_i_sub = pid_i % GROUP_SIZE_I
    
    # Calculate memory offsets for superblocks and within superblocks
    o_offset_sup = pid_o_sup * BLOCK_SIZE_O * GROUP_SIZE_O
    i_offset_sup = pid_i_sup * BLOCK_SIZE_I * GROUP_SIZE_I
    
    o_offset_sub = pid_o_sub * BLOCK_SIZE_O
    i_offset_sub = pid_i_sub * BLOCK_SIZE_I
    
    # Total memory offsets
    o_offset = o_offset_sup + o_offset_sub
    i_offset = i_offset_sup + i_offset_sub
    
    # Create block pointers for the output and input dimensions
    out_range = o_offset + tl.arange(0, BLOCK_SIZE_O)
    in_range = i_offset + tl.arange(0, BLOCK_SIZE_I)
    
    # Mask out elements that are outside the bounds
    out_mask = out_range < O
    in_mask = in_range < I
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate through batch dimension in blocks
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        b_start = b * BLOCK_SIZE_B
        
        # Create pointers to grad_output and input blocks
        grad_output_ptrs = grad_output_ptr + ((b_start + tl.arange(0, BLOCK_SIZE_B))[:, None] * stride_grad_output_b +
                                             out_range[None, :] * stride_grad_output_o)
        input_ptrs = input_ptr + ((b_start + tl.arange(0, BLOCK_SIZE_B))[:, None] * stride_input_b +
                                 in_range[None, :] * stride_input_i)
        
        # Create masks for the batch dimension
        b_mask = (b_start + tl.arange(0, BLOCK_SIZE_B)) < B
        
        # Load grad_output and input blocks with appropriate masking
        grad_output_block = tl.load(grad_output_ptrs, mask=b_mask[:, None] & out_mask[None, :], other=0.0)
        input_block = tl.load(input_ptrs, mask=b_mask[:, None] & in_mask[None, :], other=0.0)
        
        # Compute matrix multiplication: grad_weight = grad_output.T @ input
        acc += tl.dot(tl.trans(grad_output_block), input_block)
    
    # Store the results
    grad_weight_ptrs = grad_weight_ptr + out_range[:, None] * stride_grad_weight_o + in_range[None, :] * stride_grad_weight_i
    tl.store(grad_weight_ptrs, acc, mask=out_mask[:, None] & in_mask[None, :])

class TritonLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Forward pass for the linear layer: output = input @ weight.T
        
        Args:
            ctx: Context object to save tensors for backward pass
            input (torch.Tensor): Input tensor. [shape: (B, I)]
            weight (torch.Tensor): Weight tensor. [shape: (O, I)]
            
        Returns:
            torch.Tensor: Output tensor. [shape: (B, O)]
            
        Explanation:
            1. Save input and weight tensors for backward pass.
            2. Get tensor dimensions and verify shape compatibility.
            3. Allocate output tensor.
            4. Calculate tensor strides for efficient memory access.
            5. Launch Triton kernel to perform matrix multiplication.
            6. Return the computed output tensor.
        """
        ctx.save_for_backward(input, weight)
        
        # Get tensor dimensions
        B, I = input.shape
        O, I_w = weight.shape
        assert I == I_w, f"Input feature dimension mismatch: {I} vs {I_w}"
        
        # Allocate output tensor
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        # Calculate strides for tensor elements
        input_stride_b, input_stride_i = input.stride()
        weight_stride_o, weight_stride_i = weight.stride()
        output_stride_b, output_stride_o = output.stride()
        
        # Launch Triton kernel for forward pass
        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
        )
        
        forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input_stride_b, input_stride_i,
            weight_stride_o, weight_stride_i,
            output_stride_b, output_stride_o,
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the linear layer.
        
        Args:
            ctx: Context object with saved tensors
            grad_output (torch.Tensor): Gradient with respect to the output tensor. [shape: (B, O)]
            
        Returns:
            tuple: Gradients with respect to the input and weight tensors. [shapes: (B, I), (O, I)]
            
        Explanation:
            1. Retrieve saved tensors from context.
            2. Get tensor dimensions.
            3. Allocate tensors for gradients.
            4. Calculate tensor strides for efficient memory access.
            5. Launch kernel to compute gradient with respect to input.
            6. Launch kernel to compute gradient with respect to weight.
            7. Return computed gradients.
        """
        input, weight = ctx.saved_tensors
        
        # Get tensor dimensions
        B, I = input.shape
        O, _ = weight.shape
        
        # Allocate tensors for gradients
        grad_input = torch.empty_like(input)
        grad_weight = torch.empty_like(weight)
        
        # Calculate strides for tensor elements
        grad_output_stride_b, grad_output_stride_o = grad_output.stride()
        input_stride_b, input_stride_i = input.stride()
        weight_stride_o, weight_stride_i = weight.stride()
        grad_input_stride_b, grad_input_stride_i = grad_input.stride()
        grad_weight_stride_o, grad_weight_stride_i = grad_weight.stride()
        
        # Compute gradient with respect to input: grad_input = grad_output @ weight
        grid_input = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        
        backward_input_kernel[grid_input](
            grad_output, weight, grad_input,
            B, I, O,
            grad_output_stride_b, grad_output_stride_o,
            weight_stride_o, weight_stride_i,
            grad_input_stride_b, grad_input_stride_i,
        )
        
        # Compute gradient with respect to weight: grad_weight = grad_output.T @ input
        grid_weight = lambda meta: (
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        
        backward_weight_kernel[grid_weight](
            grad_output, input, grad_weight,
            B, I, O,
            grad_output_stride_b, grad_output_stride_o,
            input_stride_b, input_stride_i,
            grad_weight_stride_o, grad_weight_stride_i,
        )
        
        return grad_input, grad_weight

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        """
        Initialize the TritonLinear module.
        
        Args:
            in_features (int): Size of each input sample
            out_features (int): Size of each output sample
            device (str): Device to place the module on
            dtype (torch.dtype): Data type of the parameters
            
        Explanation:
            1. Initialize parent class.
            2. Store feature dimensions as instance variables.
            3. Create weight parameter with specified device and data type.
            4. Initialize weights using Kaiming uniform initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Reset the module parameters using Kaiming uniform initialization.
        
        Explanation:
            1. Apply Kaiming uniform initialization to the weight parameter.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, input):
        """
        Perform the forward pass of the linear layer.
        
        Args:
            input (torch.Tensor): Input tensor. [shape: (B, in_features)]
            
        Returns:
            torch.Tensor: Output tensor. [shape: (B, out_features)]
            
        Explanation:
            1. Apply the custom TritonLinearFunction to input and weight.
            2. Return the computed output.
        """
        return TritonLinearFunction.apply(input, self.weight)

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
