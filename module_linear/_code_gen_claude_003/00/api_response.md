Timestamp: 20250331_045431

# thinking
Alright, let's break down the task:

1. I need to create a PyTorch module called `TritonLinear` that implements the functionality of `nn.Linear`, but using Triton kernels.
2. I need to implement both forward and backward passes.
3. I should use autotune for block sizes and super blocks.
4. I need to test correctness and performance using the provided code.

First, let's understand what `nn.Linear` does:
- It applies a linear transformation to the incoming data: y = xA^T + b
- In this case, we're implementing it without bias (bias=False), so it's just y = xA^T

For the Triton implementation, I'll need:
1. A forward kernel that computes y = xA^T
2. A backward kernel that computes:
   - Gradient with respect to the input: dx = dy * A
   - Gradient with respect to the weight: dw = x^T * dy

Let's implement the `linear_triton.py` file:

```python
# Import necessary libraries
import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple, Union, List
```

Next, I'll define the forward kernel:

```python
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_B': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_O': 16,
            'GROUP_SIZE_B': 1,
        }, num_stages=1, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 2,
        }, num_stages=1, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 4,
        }, num_stages=1, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 128,
            'BLOCK_SIZE_I': 128,
            'BLOCK_SIZE_O': 128,
            'GROUP_SIZE_B': 8,
        }, num_stages=1, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_forward_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, output_ptr,
    # Matrix dimensions
    B, I, O,
    # The stride variables specify how much you need to increase the pointer to get to the next row / column
    stride_xb, stride_xi,
    stride_wo, stride_wi,
    stride_ob, stride_oi,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the forward pass of a linear layer: y = x @ w.t()
    
    Args:
        x_ptr (tensor pointer): Pointer to the input tensor [B, I]
        w_ptr (tensor pointer): Pointer to the weight tensor [O, I]
        output_ptr (tensor pointer): Pointer to the output tensor [B, O]
        B (int): Batch size
        I (int): Input dimension
        O (int): Output dimension
        stride_xb (int): Stride for moving along the batch dimension of x
        stride_xi (int): Stride for moving along the input dimension of x
        stride_wo (int): Stride for moving along the output dimension of w
        stride_wi (int): Stride for moving along the input dimension of w
        stride_ob (int): Stride for moving along the batch dimension of output
        stride_oi (int): Stride for moving along the output dimension of output
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension
    """
    # -----------------------------------------------------------
    # Map program IDs to blocks of the matrix multiplication
    # -----------------------------------------------------------
    # Program ID
    pid_b = tl.program_id(axis=0)
    pid_o = tl.program_id(axis=1)
    
    # Define block start indices
    offs_b_start = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    # Create offset arrays for each dimension
    # B offsets with shape (BLOCK_SIZE_B * GROUP_SIZE_B, 1)
    offs_b = offs_b_start + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
    # We use tl.arange to create a 1D array of offsets
    # I offsets with shape (1, BLOCK_SIZE_I)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    
    # Create a mask to handle bounds checking for output
    mask_o = offs_o < O
    
    # Iterate through the batch blocks
    for group_idx in range(GROUP_SIZE_B):
        # Initialize accumulator for this block
        acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
        # Select the current batch block
        current_offs_b = offs_b[group_idx * BLOCK_SIZE_B:(group_idx + 1) * BLOCK_SIZE_B]
        # Create a mask to handle bounds checking for batch
        mask_b = current_offs_b < B
        
        # Iterate through the reduction dimension (I) blocks
        for i in range(0, I, BLOCK_SIZE_I):
            # Create a mask to handle bounds checking for input
            mask_i = (i + offs_i) < I
            
            # Calculate current input indices
            x_ptrs = x_ptr + (current_offs_b[:, None] * stride_xb + (i + offs_i[None, :]) * stride_xi)
            # Load input block using the mask
            x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_i[None, :], other=0.0)
            
            # Calculate current weight indices
            w_ptrs = w_ptr + (offs_o[None, :] * stride_wo + (i + offs_i[:, None]) * stride_wi)
            # Load weight block using the mask
            w = tl.load(w_ptrs, mask=mask_i[:, None] & mask_o[None, :], other=0.0)
            
            # Perform matrix multiplication for this block: [B, I] @ [I, O] = [B, O]
            # For each output position, we need to multiply the corresponding input and weight,
            # and add to the accumulator
            acc += tl.dot(x, w)
        
        # Store the result to the output matrix
        output_ptrs = output_ptr + (current_offs_b[:, None] * stride_ob + offs_o[None, :] * stride_oi)
        tl.store(output_ptrs, acc, mask=mask_b[:, None] & mask_o[None, :])
```

Now let's implement the backward kernel for input gradients (dx):

```python
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_B': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_O': 16,
            'GROUP_SIZE_B': 1,
        }, num_stages=1, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 2,
        }, num_stages=1, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 4,
        }, num_stages=1, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 128,
            'BLOCK_SIZE_I': 128,
            'BLOCK_SIZE_O': 128,
            'GROUP_SIZE_B': 8,
        }, num_stages=1, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_dx_kernel(
    # Pointers to matrices
    dout_ptr, w_ptr, dx_ptr,
    # Matrix dimensions
    B, O, I,
    # The stride variables specify how much you need to increase the pointer to get to the next row / column
    stride_doutb, stride_douto,
    stride_wo, stride_wi,
    stride_dxb, stride_dxi,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the backward pass for the input: dx = dout @ w
    
    Args:
        dout_ptr (tensor pointer): Pointer to the output gradient tensor [B, O]
        w_ptr (tensor pointer): Pointer to the weight tensor [O, I]
        dx_ptr (tensor pointer): Pointer to the input gradient tensor [B, I]
        B (int): Batch size
        O (int): Output dimension
        I (int): Input dimension
        stride_doutb (int): Stride for moving along the batch dimension of dout
        stride_douto (int): Stride for moving along the output dimension of dout
        stride_wo (int): Stride for moving along the output dimension of w
        stride_wi (int): Stride for moving along the input dimension of w
        stride_dxb (int): Stride for moving along the batch dimension of dx
        stride_dxi (int): Stride for moving along the input dimension of dx
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension
    """
    # -----------------------------------------------------------
    # Map program IDs to blocks of the matrix multiplication
    # -----------------------------------------------------------
    # Program ID
    pid_b = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)
    
    # Define block start indices
    offs_b_start = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Create offset arrays for each dimension
    # B offsets with shape (BLOCK_SIZE_B * GROUP_SIZE_B, 1)
    offs_b = offs_b_start + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
    # O offsets with shape (1, BLOCK_SIZE_O)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    
    # Create a mask to handle bounds checking for input
    mask_i = offs_i < I
    
    # Iterate through the batch blocks
    for group_idx in range(GROUP_SIZE_B):
        # Initialize accumulator for this block
        acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
        # Select the current batch block
        current_offs_b = offs_b[group_idx * BLOCK_SIZE_B:(group_idx + 1) * BLOCK_SIZE_B]
        # Create a mask to handle bounds checking for batch
        mask_b = current_offs_b < B
        
        # Iterate through the reduction dimension (O) blocks
        for o in range(0, O, BLOCK_SIZE_O):
            # Create a mask to handle bounds checking for output
            mask_o = (o + offs_o) < O
            
            # Calculate current dout indices
            dout_ptrs = dout_ptr + (current_offs_b[:, None] * stride_doutb + (o + offs_o[None, :]) * stride_douto)
            # Load dout block using the mask
            dout = tl.load(dout_ptrs, mask=mask_b[:, None] & mask_o[None, :], other=0.0)
            
            # Calculate current weight indices - transpose of weight matrix
            w_ptrs = w_ptr + ((o + offs_o[:, None]) * stride_wo + offs_i[None, :] * stride_wi)
            # Load transposed weight block using the mask
            w = tl.load(w_ptrs, mask=mask_o[:, None] & mask_i[None, :], other=0.0)
            
            # Perform matrix multiplication for this block: [B, O] @ [O, I] = [B, I]
            # Note that the weight matrix is effectively transposed in this operation
            acc += tl.dot(dout, w)
        
        # Store the result to the dx matrix
        dx_ptrs = dx_ptr + (current_offs_b[:, None] * stride_dxb + offs_i[None, :] * stride_dxi)
        tl.store(dx_ptrs, acc, mask=mask_b[:, None] & mask_i[None, :])
```

Now let's implement the backward kernel for weight gradients (dw):

```python
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_O': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_B': 16,
            'GROUP_SIZE_O': 1,
        }, num_stages=1, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 32,
            'GROUP_SIZE_O': 2,
        }, num_stages=1, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_O': 64,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_B': 64,
            'GROUP_SIZE_O': 4,
        }, num_stages=1, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_O': 128,
            'BLOCK_SIZE_I': 128,
            'BLOCK_SIZE_B': 128,
            'GROUP_SIZE_O': 8,
        }, num_stages=1, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_dw_kernel(
    # Pointers to matrices
    x_ptr, dout_ptr, dw_ptr,
    # Matrix dimensions
    B, I, O,
    # The stride variables specify how much you need to increase the pointer to get to the next row / column
    stride_xb, stride_xi,
    stride_doutb, stride_douto,
    stride_dwo, stride_dwi,
    # Meta-parameters
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute the backward pass for the weight: dw = dout.t() @ x
    
    Args:
        x_ptr (tensor pointer): Pointer to the input tensor [B, I]
        dout_ptr (tensor pointer): Pointer to the output gradient tensor [B, O]
        dw_ptr (tensor pointer): Pointer to the weight gradient tensor [O, I]
        B (int): Batch size
        I (int): Input dimension
        O (int): Output dimension
        stride_xb (int): Stride for moving along the batch dimension of x
        stride_xi (int): Stride for moving along the input dimension of x
        stride_doutb (int): Stride for moving along the batch dimension of dout
        stride_douto (int): Stride for moving along the output dimension of dout
        stride_dwo (int): Stride for moving along the output dimension of dw
        stride_dwi (int): Stride for moving along the input dimension of dw
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        GROUP_SIZE_O (tl.constexpr): Group size for output dimension
    """
    # -----------------------------------------------------------
    # Map program IDs to blocks of the matrix multiplication
    # -----------------------------------------------------------
    # Program ID
    pid_o = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)
    
    # Define block start indices
    offs_o_start = pid_o * BLOCK_SIZE_O * GROUP_SIZE_O
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Create offset arrays for each dimension
    # O offsets with shape (BLOCK_SIZE_O * GROUP_SIZE_O, 1)
    offs_o = offs_o_start + tl.arange(0, BLOCK_SIZE_O * GROUP_SIZE_O)
    # B offsets with shape (1, BLOCK_SIZE_B)
    offs_b = tl.arange(0, BLOCK_SIZE_B)
    
    # Create a mask to handle bounds checking for input
    mask_i = offs_i < I
    
    # Iterate through the output blocks
    for group_idx in range(GROUP_SIZE_O):
        # Initialize accumulator for this block
        acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
        # Select the current output block
        current_offs_o = offs_o[group_idx * BLOCK_SIZE_O:(group_idx + 1) * BLOCK_SIZE_O]
        # Create a mask to handle bounds checking for output
        mask_o = current_offs_o < O
        
        # Iterate through the reduction dimension (B) blocks
        for b in range(0, B, BLOCK_SIZE_B):
            # Create a mask to handle bounds checking for batch
            mask_b = (b + offs_b) < B
            
            # Calculate current x indices - we need to transpose x for weight gradient
            x_ptrs = x_ptr + ((b + offs_b)[:, None] * stride_xb + offs_i[None, :] * stride_xi)
            # Load x block using the mask
            x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_i[None, :], other=0.0)
            
            # Calculate current dout indices
            dout_ptrs = dout_ptr + ((b + offs_b)[None, :] * stride_doutb + current_offs_o[:, None] * stride_douto)
            # Load dout block using the mask
            dout = tl.load(dout_ptrs, mask=mask_o[:, None] & mask_b[None, :], other=0.0)
            
            # Perform matrix multiplication for this block: [O, B] @ [B, I] = [O, I]
            # We compute dout.T @ x for weight gradient
            acc += tl.dot(dout, x)
        
        # Store the result to the dw matrix
        dw_ptrs = dw_ptr + (current_offs_o[:, None] * stride_dwo + offs_i[None, :] * stride_dwi)
        tl.store(dw_ptrs, acc, mask=mask_o[:, None] & mask_i[None, :])
```

Now let's define the `TritonLinear` module:

```python
class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        """
        Triton implementation of a linear layer without bias.
        
        Args:
            in_features (int): Input features dimension
            out_features (int): Output features dimension
            device (str or torch.device): Device to store parameters
            dtype (torch.dtype): Data type of parameters
        """
        super(TritonLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialize parameters using Kaiming uniform initialization.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def get_parameter(self, name):
        """
        Get parameter by name.
        
        Args:
            name (str): Parameter name
            
        Returns:
            torch.nn.Parameter: The requested parameter
        """
        if name == 'weight':
            return self.weight
        return None
        
    def forward(self, x):
        """
        Forward pass using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_features]
        """
        batch_size = x.shape[0]
        output = torch.empty((batch_size, self.out_features), device=x.device, dtype=x.dtype)
        
        # Ensure contiguous inputs
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Calculate grid dimensions
        # We launch one block for each batch slice and each output slice
        grid = (
            triton.cdiv(batch_size, 128),  # blocks for batch dimension
            triton.cdiv(self.out_features, 128)  # blocks for output dimension
        )
        
        # Launch the forward kernel
        linear_forward_kernel[grid](
            x, weight, output,
            batch_size, self.in_features, self.out_features,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        
        return output
    
    def backward(self, x, dout):
        """
        Backward pass for custom autograd.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features]
            dout (torch.Tensor): Gradient tensor of shape [batch_size, out_features]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - dx: Gradient with respect to input, shape [batch_size, in_features]
                - dw: Gradient with respect to weight, shape [out_features, in_features]
        """
        batch_size = x.shape[0]
        
        # Ensure contiguous inputs
        x = x.contiguous()
        dout = dout.contiguous()
        
        # Compute gradient with respect to input (dx)
        dx = torch.empty((batch_size, self.in_features), device=x.device, dtype=x.dtype)
        # Calculate grid dimensions for dx computation
        dx_grid = (
            triton.cdiv(batch_size, 128),  # blocks for batch dimension
            triton.cdiv(self.in_features, 128)  # blocks for input dimension
        )
        
        # Launch the dx backward kernel
        linear_backward_dx_kernel[dx_grid](
            dout, self.weight, dx,
            batch_size, self.out_features, self.in_features,
            dout.stride(0), dout.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            dx.stride(0), dx.stride(1),
        )
        
        # Compute gradient with respect to weights (dw)
        dw = torch.empty_like(self.weight)
        # Calculate grid dimensions for dw computation
        dw_grid = (
            triton.cdiv(self.out_features, 128),  # blocks for output dimension
            triton.cdiv(self.in_features, 128)  # blocks for input dimension
        )
        
        # Launch the dw backward kernel
        linear_backward_dw_kernel[dw_grid](
            x, dout, dw,
            batch_size, self.in_features, self.out_features,
            x.stride(0), x.stride(1),
            dout.stride(0), dout.stride(1),
            dw.stride(0), dw.stride(1),
        )
        
        return dx, dw
```

Now, we need to handle autograd properly. Let's update the code:

```python
# Custom autograd function for the linear layer
class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """
        Forward pass for the linear layer.
        
        Args:
            ctx (torch.autograd.function.Context): Autograd context object
            x (torch.Tensor): Input tensor of shape [batch_size, in_features]
            weight (torch.Tensor): Weight tensor of shape [out_features, in_features]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_features]
        """
        batch_size = x.shape[0]
        out_features = weight.shape[0]
        in_features = weight.shape[1]
        
        # Ensure contiguous inputs
        x = x.contiguous()
        weight = weight.contiguous()
        
        # Save inputs for backward
        ctx.save_for_backward(x, weight)
        
        # Create output tensor
        output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
        
        # Calculate grid dimensions
        grid = (
            triton.cdiv(batch_size, 128),  # blocks for batch dimension
            triton.cdiv(out_features, 128)  # blocks for output dimension
        )
        
        # Launch the forward kernel
        linear_forward_kernel[grid](
            x, weight, output,
            batch_size, in_features, out_features,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        
        return output
    
    @staticmethod
    def backward(ctx, dout):
        """
        Backward pass for the linear layer.
        
        Args:
            ctx (torch.autograd.function.Context): Autograd context object
            dout (torch.Tensor): Gradient tensor of shape [batch_size, out_features]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - dx: Gradient with respect to input, shape [batch_size, in_features]
                - dw: Gradient with respect to weight, shape [out_features, in_features]
        """
        x, weight = ctx.saved_tensors
        batch_size = x.shape[0]
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        
        # Ensure contiguous gradients
        dout = dout.contiguous()
        
        # Compute gradient with respect to input (dx)
        dx = torch.empty((batch_size, in_features), device=x.device, dtype=x.dtype)
        # Calculate grid dimensions for dx computation
        dx_grid = (
            triton.cdiv(batch_size, 128),  # blocks for batch dimension
            triton.cdiv(in_features, 128)  # blocks for input dimension
        )
        
        # Launch the dx backward kernel
        linear_backward_dx_kernel[dx_grid](
            dout, weight, dx,
            batch_size, out_features, in_features,
            dout.stride(0), dout.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
        )
        
        # Compute gradient with respect to weights (dw)
        dw = torch.empty_like(weight)
        # Calculate grid dimensions for dw computation
        dw_grid = (
            triton.cdiv(out_features, 128),  # blocks for output dimension
            triton.cdiv(in_features, 128)  # blocks for input dimension
        )
        
        # Launch the dw backward kernel
        linear_backward_dw_kernel[dw_grid](
            x, dout, dw,
            batch_size, in_features, out_features,
            x.stride(0), x.stride(1),
            dout.stride(0), dout.stride(1),
            dw.stride(0), dw.stride(1),
        )
        
        return dx, dw
```

Now, let's update the `TritonLinear` module to use the custom autograd function:

```python
class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        """
        Triton implementation of a linear layer without bias.
        
        Args:
            in_features (int): Input features dimension
            out_features (int): Output features dimension
            device (str or torch.device): Device to store parameters
            dtype (torch.dtype): Data type of parameters
        """
        super(TritonLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialize parameters using Kaiming uniform initialization.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def get_parameter(self, name):
        """
        Get parameter by name.
        
        Args:
            name (str): Parameter name
            
        Returns:
            torch.nn.Parameter: The requested parameter
        """
        if name == 'weight':
            return self.weight
        return None
        
    def forward(self, x):
        """
        Forward pass using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_features]
        """
        return TritonLinearFunction.apply(x, self.weight)
```

Wait, I just noticed a problem. The autotune code for the kernels is a bit naive. Let's adjust the autotuning parameters to better match the problem:

```python
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 1,
        }, num_stages=3, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_O': 16,
            'GROUP_SIZE_B': 1,
        }, num_stages=1, num_warps=2),
    ],
    key=['B', 'I', 'O'],
)
```

Actually, let's use the same pattern for all kernels, but adjust the key variables to match the different dimensions used in each kernel.

I'll put it all together into one file now:

```python
# filename: linear_triton.py
import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 1,
        }, num_stages=3, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_O': 16,
            'GROUP_SIZE_B': 1,
        }, num_stages=1, num_warps=2),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_forward_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, output_ptr,
    # Matrix dimensions
    B, I, O,
    # The stride variables specify how much you need to increase the pointer to get to the next row / column
    stride_xb, stride_xi,
    stride_wo, stride_wi,
    stride_ob, stride_oi,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the forward pass of a linear layer: y = x @ w.t()
    
    Args:
        x_ptr (tensor pointer): Pointer to the input tensor [B, I]
        w_ptr (tensor pointer): Pointer to the weight tensor [O, I]
        output_ptr (tensor pointer): Pointer to the output tensor [B, O]
        B (int): Batch size
        I (int): Input dimension
        O (int): Output dimension
        stride_xb (int): Stride for moving along the batch dimension of x
        stride_xi (int): Stride for moving along the input dimension of x
        stride_wo (int): Stride for moving along the output dimension of w
        stride_wi (int): Stride for moving along the input dimension of w
        stride_ob (int): Stride for moving along the batch dimension of output
        stride_oi (int): Stride for moving along the output dimension of output
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension
    """
    # -----------------------------------------------------------
    # Map program IDs to blocks of the matrix multiplication
    # -----------------------------------------------------------
    # Program ID
    pid_b = tl.program_id(axis=0)
    pid_o = tl.program_id(axis=1)
    
    # Define block start indices
    offs_b_start = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    # Create offset arrays for each dimension
    # B offsets with shape (BLOCK_SIZE_B * GROUP_SIZE_B, 1)
    offs_b = offs_b_start + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
    # We use tl.arange to create a 1D array of offsets
    # I offsets with shape (1, BLOCK_SIZE_I)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    
    # Create a mask to handle bounds checking for output
    mask_o = offs_o < O
    
    # Iterate through the batch blocks
    for group_idx in range(GROUP_SIZE_B):
        # Initialize accumulator for this block
        acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
        # Select the current batch block
        current_offs_b = offs_b[group_idx * BLOCK_SIZE_B:(group_idx + 1) * BLOCK_SIZE_B]
        # Create a mask to handle bounds checking for batch
        mask_b = current_offs_b < B
        
        # Iterate through the reduction dimension (I) blocks
        for i in range(0, I, BLOCK_SIZE_I):
            # Create a mask to handle bounds checking for input
            mask_i = (i + offs_i) < I
            
            # Calculate current input indices
            x_ptrs = x_ptr + (current_offs_b[:, None] * stride_xb + (i + offs_i[None, :]) * stride_xi)
            # Load input block using the mask
            x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_i[None, :], other=0.0)
            
            # Calculate current weight indices
            w_ptrs = w_ptr + (offs_o[None, :] * stride_wo + (i + offs_i[:, None]) * stride_wi)
            # Load weight block using the mask
            w = tl.load(w_ptrs, mask=mask_i[:, None] & mask_o[None, :], other=0.0)
            
            # Perform matrix multiplication for this block: [B, I] @ [I, O] = [B, O]
            # For each output position, we need to multiply the corresponding input and weight,
            # and add to the accumulator
            acc += tl.dot(x, w)
        
        # Store the result to the output matrix
        output_ptrs = output_ptr + (current_offs_b[:, None] * stride_ob + offs_o[None, :] * stride_oi)
        tl.store(output_ptrs, acc, mask=mask_b[:, None] & mask_o[None, :])

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 1,
        }, num_stages=3, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_O': 16,
            'GROUP_SIZE_B': 1,
        }, num_stages=1, num_warps=2),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_dx_kernel(
    # Pointers to matrices
    dout_ptr, w_ptr, dx_ptr,
    # Matrix dimensions
    B, O, I,
    # The stride variables specify how much you need to increase the pointer to get to the next row / column
    stride_doutb, stride_douto,
    stride_wo, stride_wi,
    stride_dxb, stride_dxi,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the backward pass for the input: dx = dout @ w
    
    Args:
        dout_ptr (tensor pointer): Pointer to the output gradient tensor [B, O]
        w_ptr (tensor pointer): Pointer to the weight tensor [O, I]
        dx_ptr (tensor pointer): Pointer to the input gradient tensor [B, I]
        B (int): Batch size
        O (int): Output dimension
        I (int): Input dimension
        stride_doutb (int): Stride for moving along the batch dimension of dout
        stride_douto (int): Stride for moving along the output dimension of dout
        stride_wo (int): Stride for moving along the output dimension of w
        stride_wi (int): Stride for moving along the input dimension of w
        stride_dxb (int): Stride for moving along the batch dimension of dx
        stride_dxi (int): Stride for moving along the input dimension of dx
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension
    """
    # -----------------------------------------------------------
    # Map program IDs to blocks of the matrix multiplication
    # -----------------------------------------------------------
    # Program ID
    pid_b = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)
    
    # Define block start indices
    offs_b_start = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Create offset arrays for each dimension
    # B offsets with shape (BLOCK_SIZE_B * GROUP_SIZE_B, 1)
    offs_b = offs_b_start + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
    # O offsets with shape (1, BLOCK_SIZE_O)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    
    # Create a mask to handle bounds checking for input
    mask_i = offs_i < I
    
    # Iterate through the batch blocks
    for group_idx in range(GROUP_SIZE_B):
        # Initialize accumulator for this block
        acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
        # Select the current batch block
        current_offs_b = offs_b[group_idx * BLOCK_SIZE_B:(group_idx + 1) * BLOCK_SIZE_B]
        # Create a mask to handle bounds checking for batch
        mask_b = current_offs_b < B
        
        # Iterate through the reduction dimension (O) blocks
        for o in range(0, O, BLOCK_SIZE_O):
            # Create a mask to handle bounds checking for output
            mask_o = (o + offs_o) < O
            
            # Calculate current dout indices
            dout_ptrs = dout_ptr + (current_offs_b[:, None] * stride_doutb + (o + offs_o[None, :]) * stride_douto)
            # Load dout block using the mask
            dout = tl.load(dout_ptrs, mask=mask_b[:, None] & mask_o[None, :], other=0.0)
            
            # Calculate current weight indices - transpose of weight matrix
            w_ptrs = w_ptr + ((o + offs_o[:, None]) * stride_wo + offs_i[None, :] * stride_wi)
            # Load transposed weight block using the mask
            w = tl.load(w_ptrs, mask=mask_o[:, None] & mask_i[None, :], other=0.0)
            
            # Perform matrix multiplication for this block: [B, O] @ [O, I] = [B, I]
            # Note that the weight matrix is effectively transposed in this operation
            acc += tl.dot(dout, w)
        
        # Store the result to the dx matrix
        dx_ptrs = dx_ptr + (current_offs_b[:, None] * stride_dxb + offs_i[None, :] * stride_dxi)
        tl.store(dx_ptrs, acc, mask=mask_b[:, None] & mask_i[None, :])

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 32,
            'GROUP_SIZE_O': 1,
        }, num_stages=3, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 64,
            'GROUP_SIZE_O': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 32,
            'GROUP_SIZE_O': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 64,
            'GROUP_SIZE_O': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_B': 32,
            'GROUP_SIZE_O': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_B': 64,
            'GROUP_SIZE_O': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_B': 16,
            'GROUP_SIZE_O': 1,
        }, num_stages=1, num_warps=2),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_dw_kernel(
    # Pointers to matrices
    x_ptr, dout_ptr, dw_ptr,
    # Matrix dimensions
    B, I, O,
    # The stride variables specify how much you need to increase the pointer to get to the next row / column
    stride_xb, stride_xi,
    stride_doutb, stride_douto,
    stride_dwo, stride_dwi,
    # Meta-parameters
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute the backward pass for the weight: dw = dout.t() @ x
    
    Args:
        x_ptr (tensor pointer): Pointer to the input tensor [B, I]
        dout_ptr (tensor pointer): Pointer to the output gradient tensor [B, O]
        dw_ptr (tensor pointer): Pointer to the weight gradient tensor [O, I]
        B (int): Batch size
        I (int): Input dimension
        O (int): Output dimension
        stride_xb (int): Stride for moving along the batch dimension of x
        stride_xi (int): Stride for moving along the input dimension of x
        stride_doutb (int): Stride for moving along the batch dimension of dout
        stride_douto (int): Stride for moving along the output dimension of dout
        stride_dwo (int): Stride for moving along the output dimension of dw
        stride_dwi (int): Stride for moving along the input dimension of dw
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        GROUP_SIZE_O (tl.constexpr): Group size for output dimension
    """
    # -----------------------------------------------------------
    # Map program IDs to blocks of the matrix multiplication
    # -----------------------------------------------------------
    # Program ID
    pid_o = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)
    
    # Define block start indices
    offs_o_start = pid_o * BLOCK_SIZE_O * GROUP_SIZE_O
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Create offset arrays for each dimension
    # O offsets with shape (BLOCK_SIZE_O * GROUP_SIZE_O, 1)
    offs_o = offs_o_start + tl.arange(0, BLOCK_SIZE_O * GROUP_SIZE_O)
    # B offsets with shape (1, BLOCK_SIZE_B)
    offs_b = tl.arange(0, BLOCK_SIZE_B)
    
    # Create a mask to handle bounds checking for input
    mask_i = offs_i < I
    
    # Iterate through the output blocks
    for group_idx in range(GROUP_SIZE_O):
        # Initialize accumulator for this block
        acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
        # Select the current output block
        current_offs_o = offs_o[group_idx * BLOCK_SIZE_O:(group_idx + 1) * BLOCK_SIZE_O]
        # Create a mask to handle bounds checking for output
        mask_o = current_offs_o < O
        
        # Iterate through the reduction dimension (B) blocks
        for b in range(0, B, BLOCK_SIZE_B):
            # Create a mask to handle bounds checking for batch
            mask_b = (b + offs_b) < B
            
            # Calculate current x indices - we need to transpose x for weight gradient
            x_ptrs = x_ptr + ((b + offs_b)[:, None] * stride_xb + offs_i[None, :] * stride_xi)
            # Load x block using the mask
            x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_i[None, :], other=0.0)
            
            # Calculate current dout indices
            dout_ptrs = dout_ptr + ((b + offs_b)[None, :] * stride_doutb + current_offs_o[:, None] * stride_douto)
            # Load dout block using the mask
            dout = tl.load(dout_ptrs, mask=mask_o[:, None] & mask_b[None, :], other=0.0)
            
            # Perform matrix multiplication for this block: [O, B] @ [B, I] = [O, I]
            # We compute dout.T @ x for weight gradient
            acc += tl.dot(dout, x)
        
        # Store the result to the dw matrix
        dw_ptrs = dw_ptr + (current_offs_o[:, None] * stride_dwo + offs_i[None, :] * stride_dwi)
        tl.store(dw_ptrs, acc, mask=mask_o[:, None] & mask_i[None, :])

# Custom autograd function for the linear layer
class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """
        Forward pass for the linear layer.
        
        Args:
            ctx (torch.autograd.function.Context): Autograd context object
            x (torch.Tensor): Input tensor of shape [batch_size, in_features]
            weight (torch.Tensor): Weight tensor of shape [out_features, in_features]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_features]
        """
        batch_size = x.shape[0]
        out_features = weight.shape[0]
        in_features = weight.shape[1]
        
        # Ensure contiguous inputs
        x = x.contiguous()
        weight = weight.contiguous()
        
        # Save inputs for backward
        ctx.save_for_backward(x, weight)
        
        # Create output tensor
        output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
        
        # Calculate grid dimensions
        grid = (
            triton.cdiv(batch_size, 128),  # blocks for batch dimension
            triton.cdiv(out_features, 128)  # blocks for output dimension
        )
        
        # Launch the forward kernel
        linear_forward_kernel[grid](
            x, weight, output,
            batch_size, in_features, out_features,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        
        return output
    
    @staticmethod
    def backward(ctx, dout):
        """
        Backward pass for the linear layer.
        
        Args:
            ctx (torch.autograd.function.Context): Autograd context object
            dout (torch.Tensor): Gradient tensor of shape [batch_size, out_features]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - dx: Gradient with respect to input, shape [batch_size, in_features]
                - dw: Gradient with respect to weight, shape [out_features, in_features]
        """
        x, weight = ctx.saved_tensors
        batch_size = x.shape[0]
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        
        # Ensure contiguous gradients
        dout = dout.contiguous()
        
        # Compute gradient with respect to input (dx)
        dx = torch.empty((batch_size, in_features), device=x.device, dtype=x.dtype)
        # Calculate grid dimensions for dx computation
        dx_grid = (
            triton.cdiv(batch_size, 128),  # blocks for batch dimension
            triton.cdiv(in_features, 128)  # blocks for input dimension
        )
        
        # Launch the dx backward kernel
        linear_backward_dx_kernel[dx_grid](
            dout, weight, dx,
            batch_size, out_features, in_features,
            dout.stride(0), dout.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
        )
        
        # Compute gradient with respect to weights (dw)
        dw = torch.empty_like(weight)
        # Calculate grid dimensions for dw computation
        dw_grid = (
            triton.cdiv(out_features, 128),  # blocks for output dimension
            triton.cdiv(in_features, 128)  # blocks for input dimension
        )
        
        # Launch the dw backward kernel
        linear_backward_dw_kernel[dw_grid](
            x, dout, dw,
            batch_size, in_features, out_features,
            x.stride(0), x.stride(1),
            dout.stride(0), dout.stride(1),
            dw.stride(0), dw.stride(1),
        )
        
        return dx, dw

class TritonLinear(torch.nn.Module):
    """
    Triton implementation of a linear layer without bias, equivalent to nn.Linear(in_features, out_features, bias=False).
    
    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        device (str or torch.device): Device to store parameters
        dtype (torch.dtype): Data type of parameters
        
    Attributes:
        weight (torch.nn.Parameter): The learnable weights of the module of shape (out_features, in_features)
    """
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super(TritonLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialize parameters using Kaiming uniform initialization.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def get_parameter(self, name):
        """
        Get parameter by name.
        
        Args:
            name (str): Parameter name
            
        Returns:
            torch.nn.Parameter: The requested parameter
        """
        if name == 'weight':
            return self.weight
        return None
        
    def forward(self, x):
        """
        Forward pass using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_features]
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
```

Let me check this implementation to make sure everything is working as expected:

1. I've implemented a `TritonLinear` module with:
   - A forward pass that uses a Triton kernel to compute `y = x @ w.t()`
   - A backward pass that computes:
     - `dx = dout @ w` for the input gradient
     - `dw = dout.t() @ x` for the weight gradient
   - Use of autotuning for block sizes
   - Use of super blocks via `GROUP_SIZE_` parameters

2. The implementation follows PyTorch's autograd properly by implementing a custom autograd Function.

3. I've included the test code from the problem statement to test correctness and benchmark performance.

I think the implementation looks good. Let me make sure the docstrings match the required format from the problem statement:

1. The problem requires docstrings with:
   - Description
   - Args (with type hints and shapes for tensors)
   - Returns (with type hints and shapes)
   - Explanation (numbered list of steps)

Let me update docstrings to match this format:

Alright, let me write the final version of the code with improved docstrings that match the requirements:

```python
# filename: linear_triton.py
import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 1,
        }, num_stages=3, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_O': 16,
            'GROUP_SIZE_B': 1,
        }, num_stages=1, num_warps=2),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_forward_kernel(
    x_ptr, w_ptr, output_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_wo, stride_wi,
    stride_ob, stride_oi,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the forward pass of a linear layer: y = x @ w.t()

    Args:
        x_ptr (ptr): Pointer to the input tensor. [B, I]
        w_ptr (ptr): Pointer to the weight tensor. [O, I]
        output_ptr (ptr): Pointer to the output tensor. [B, O]
        B (int): Batch size.
        I (int): Input dimension.
        O (int): Output dimension.
        stride_xb (int): Stride for batch dimension of x.
        stride_xi (int): Stride for input dimension of x.
        stride_wo (int): Stride for output dimension of w.
        stride_wi (int): Stride for input dimension of w.
        stride_ob (int): Stride for batch dimension of output.
        stride_oi (int): Stride for output dimension of output.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension.

    Returns:
        None: Results are stored in output_ptr.

    Explanation:
        1. Map program IDs to blocks of the matrix multiplication.
        2. Initialize offsets and masks for boundary checking.
        3. Process multiple batch blocks using GROUP_SIZE_B.
        4. For each batch block, initialize accumulator to zeros.
        5. Iterate through input dimension in blocks of BLOCK_SIZE_I.
        6. Load blocks of input and weight matrices.
        7. Perform matrix multiplication using tl.dot.
        8. Store accumulated results to output tensor.
    """
    # 1. Map program IDs to blocks of the matrix multiplication
    pid_b = tl.program_id(axis=0)
    pid_o = tl.program_id(axis=1)
    
    # 2. Initialize offsets and masks
    offs_b_start = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_b = offs_b_start + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    mask_o = offs_o < O
    
    # 3. Process multiple batch blocks using GROUP_SIZE_B
    for group_idx in range(GROUP_SIZE_B):
        # 4. Initialize accumulator to zeros
        acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
        current_offs_b = offs_b[group_idx * BLOCK_SIZE_B:(group_idx + 1) * BLOCK_SIZE_B]
        mask_b = current_offs_b < B
        
        # 5. Iterate through input dimension in blocks
        for i in range(0, I, BLOCK_SIZE_I):
            mask_i = (i + offs_i) < I
            
            # 6. Load blocks of input and weight matrices
            x_ptrs = x_ptr + (current_offs_b[:, None] * stride_xb + (i + offs_i[None, :]) * stride_xi)
            x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_i[None, :], other=0.0)
            
            w_ptrs = w_ptr + (offs_o[None, :] * stride_wo + (i + offs_i[:, None]) * stride_wi)
            w = tl.load(w_ptrs, mask=mask_i[:, None] & mask_o[None, :], other=0.0)
            
            # 7. Perform matrix multiplication
            acc += tl.dot(x, w)
        
        # 8. Store results to output tensor
        output_ptrs = output_ptr + (current_offs_b[:, None] * stride_ob + offs_o[None, :] * stride_oi)
        tl.store(output_ptrs, acc, mask=mask_b[:, None] & mask_o[None, :])

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 1,
        }, num_stages=3, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_O': 16,
            'GROUP_SIZE_B': 1,
        }, num_stages=1, num_warps=2),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_dx_kernel(
    dout_ptr, w_ptr, dx_ptr,
    B, O, I,
    stride_doutb, stride_douto,
    stride_wo, stride_wi,
    stride_dxb, stride_dxi,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the backward pass for the input gradient: dx = dout @ w

    Args:
        dout_ptr (ptr): Pointer to the output gradient tensor. [B, O]
        w_ptr (ptr): Pointer to the weight tensor. [O, I]
        dx_ptr (ptr): Pointer to the input gradient tensor. [B, I]
        B (int): Batch size.
        O (int): Output dimension.
        I (int): Input dimension.
        stride_doutb (int): Stride for batch dimension of dout.
        stride_douto (int): Stride for output dimension of dout.
        stride_wo (int): Stride for output dimension of w.
        stride_wi (int): Stride for input dimension of w.
        stride_dxb (int): Stride for batch dimension of dx.
        stride_dxi (int): Stride for input dimension of dx.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension.

    Returns:
        None: Results are stored in dx_ptr.

    Explanation:
        1. Map program IDs to blocks of the matrix multiplication.
        2. Initialize offsets and masks for boundary checking.
        3. Process multiple batch blocks using GROUP_SIZE_B.
        4. For each batch block, initialize accumulator to zeros.
        5. Iterate through output dimension in blocks of BLOCK_SIZE_O.
        6. Load blocks of dout and weight matrices.
        7. Perform matrix multiplication using tl.dot to compute input gradients.
        8. Store accumulated results to dx tensor.
    """
    # 1. Map program IDs to blocks
    pid_b = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)
    
    # 2. Initialize offsets and masks
    offs_b_start = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = offs_b_start + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    mask_i = offs_i < I
    
    # 3. Process multiple batch blocks
    for group_idx in range(GROUP_SIZE_B):
        # 4. Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
        current_offs_b = offs_b[group_idx * BLOCK_SIZE_B:(group_idx + 1) * BLOCK_SIZE_B]
        mask_b = current_offs_b < B
        
        # 5. Iterate through output dimension
        for o in range(0, O, BLOCK_SIZE_O):
            mask_o = (o + offs_o) < O
            
            # 6. Load blocks of dout and weight
            dout_ptrs = dout_ptr + (current_offs_b[:, None] * stride_doutb + (o + offs_o[None, :]) * stride_douto)
            dout = tl.load(dout_ptrs, mask=mask_b[:, None] & mask_o[None, :], other=0.0)
            
            w_ptrs = w_ptr + ((o + offs_o[:, None]) * stride_wo + offs_i[None, :] * stride_wi)
            w = tl.load(w_ptrs, mask=mask_o[:, None] & mask_i[None, :], other=0.0)
            
            # 7. Compute input gradients
            acc += tl.dot(dout, w)
        
        # 8. Store results
        dx_ptrs = dx_ptr + (current_offs_b[:, None] * stride_dxb + offs_i[None, :] * stride_dxi)
        tl.store(dx_ptrs, acc, mask=mask_b[:, None] & mask_i[None, :])

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 32,
            'GROUP_SIZE_O': 1,
        }, num_stages=3, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 64,
            'GROUP_SIZE_O': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 32,
            'GROUP_SIZE_O': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 64,
            'GROUP_SIZE_O': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_B': 32,
            'GROUP_SIZE_O': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_B': 64,
            'GROUP_SIZE_O': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_B': 16,
            'GROUP_SIZE_O': 1,
        }, num_stages=1, num_warps=2),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_dw_kernel(
    x_ptr, dout_ptr, dw_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_doutb, stride_douto,
    stride_dwo, stride_dwi,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute the backward pass for the weight gradient: dw = dout.t() @ x

    Args:
        x_ptr (ptr): Pointer to the input tensor. [B, I]
        dout_ptr (ptr): Pointer to the output gradient tensor. [B, O]
        dw_ptr (ptr): Pointer to the weight gradient tensor. [O, I]
        B (int): Batch size.
        I (int): Input dimension.
        O (int): Output dimension.
        stride_xb (int): Stride for batch dimension of x.
        stride_xi (int): Stride for input dimension of x.
        stride_doutb (int): Stride for batch dimension of dout.
        stride_douto (int): Stride for output dimension of dout.
        stride_dwo (int): Stride for output dimension of dw.
        stride_dwi (int): Stride for input dimension of dw.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        GROUP_SIZE_O (tl.constexpr): Group size for output dimension.

    Returns:
        None: Results are stored in dw_ptr.

    Explanation:
        1. Map program IDs to blocks of the matrix multiplication.
        2. Initialize offsets and masks for boundary checking.
        3. Process multiple output blocks using GROUP_SIZE_O.
        4. For each output block, initialize accumulator to zeros.
        5. Iterate through batch dimension in blocks of BLOCK_SIZE_B.
        6. Load blocks of x and dout matrices.
        7. Perform matrix multiplication using tl.dot to compute weight gradients.
        8. Store accumulated results to dw tensor.
    """
    # 1. Map program IDs to blocks
    pid_o = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)
    
    # 2. Initialize offsets and masks
    offs_o_start = pid_o * BLOCK_SIZE_O * GROUP_SIZE_O
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = offs_o_start + tl.arange(0, BLOCK_SIZE_O * GROUP_SIZE_O)
    offs_b = tl.arange(0, BLOCK_SIZE_B)
    mask_i = offs_i < I
    
    # 3. Process multiple output blocks
    for group_idx in range(GROUP_SIZE_O):
        # 4. Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
        current_offs_o = offs_o[group_idx * BLOCK_SIZE_O:(group_idx + 1) * BLOCK_SIZE_O]
        mask_o = current_offs_o < O
        
        # 5. Iterate through batch dimension
        for b in range(0, B, BLOCK_SIZE_B):
            mask_b = (b + offs_b) < B
            
            # 6. Load blocks of x and dout
            x_ptrs = x_ptr + ((b + offs_b)[:, None] * stride_xb + offs_i[None, :] * stride_xi)
            x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_i[None, :], other=0.0)
            
            dout_ptrs = dout_ptr + ((b + offs_b)[None, :] * stride_doutb + current_offs_o[:, None] * stride_douto)
            dout = tl.load(dout_ptrs, mask=mask_o[:, None] & mask_b[None, :], other=0.0)
            
            # 7. Compute weight gradients
            acc += tl.dot(dout, x)
        
        # 8. Store results
        dw_ptrs = dw_ptr + (current_offs_o[:, None] * stride_dwo + offs_i[None, :] * stride_dwi)
        tl.store(dw_ptrs, acc, mask=mask_o[:, None] & mask_i[None, :])

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """
        Forward pass for the linear layer.

        Args:
            ctx (torch.autograd.function.Context): Autograd context object.
            x (torch.Tensor): Input tensor. [batch_size, in_features]
            weight (torch.Tensor): Weight tensor. [out_features, in_features]

        Returns:
            torch.Tensor: Output tensor. [batch_size, out_features]

        Explanation:
            1. Extract dimensions from input tensors.
            2. Ensure inputs are contiguous in memory.
            3. Save inputs for backward pass.
            4. Create output tensor.
            5. Calculate grid dimensions for kernel launch.
            6. Launch the forward Triton kernel.
            7. Return the computed output.
        """
        # 1. Extract dimensions
        batch_size = x.shape[0]
        out_features = weight.shape[0]
        in_features = weight.shape[1]
        
        # 2. Ensure contiguous inputs
        x = x.contiguous()
        weight = weight.contiguous()
        
        # 3. Save inputs for backward
        ctx.save_for_backward(x, weight)
        
        # 4. Create output tensor
        output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
        
        # 5. Calculate grid dimensions
        grid = (
            triton.cdiv(batch_size, 128),  # blocks for batch dimension
            triton.cdiv(out_features, 128)  # blocks for output dimension
        )
        
        # 6. Launch the forward kernel
        linear_forward_kernel[grid](
            x, weight, output,
            batch_size, in_features, out_features,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        
        # 7. Return output
        return output
    
    @staticmethod
    def backward(ctx, dout):
        """
        Backward pass for the linear layer.

        Args:
            ctx (torch.autograd.function.Context): Autograd context object.
            dout (torch.Tensor): Gradient tensor from upstream. [batch_size, out_features]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - dx: Gradient with respect to input. [batch_size, in_features]
                - dw: Gradient with respect to weight. [out_features, in_features]

        Explanation:
            1. Retrieve saved tensors from context.
            2. Extract dimensions from tensors.
            3. Ensure gradients are contiguous in memory.
            4. Compute gradient with respect to input (dx).
            5. Calculate grid dimensions for dx computation.
            6. Launch the dx backward kernel.
            7. Compute gradient with respect to weights (dw).
            8. Calculate grid dimensions for dw computation.
            9. Launch the dw backward kernel.
            10. Return computed gradients.
        """
        # 1. Retrieve saved tensors
        x, weight = ctx.saved_tensors
        
        # 2. Extract dimensions
        batch_size = x.shape[0]
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        
        # 3. Ensure contiguous gradients
        dout = dout.contiguous()
        
        # 4. Prepare tensor for dx
        dx = torch.empty((batch_size, in_features), device=x.device, dtype=x.dtype)
        
        # 5. Calculate grid dimensions for dx
        dx_grid = (
            triton.cdiv(batch_size, 128),  # blocks for batch dimension
            triton.cdiv(in_features, 128)  # blocks for input dimension
        )
        
        # 6. Launch the dx backward kernel
        linear_backward_dx_kernel[dx_grid](
            dout, weight, dx,
            batch_size, out_features, in_features,
            dout.stride(0), dout.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
        )
        
        # 7. Prepare tensor for dw
        dw = torch.empty_like(weight)
        
        # 8. Calculate grid dimensions for dw
        dw_grid = (
            triton.cdiv(out_features, 128),  # blocks for output dimension
            triton.cdiv(in_features, 128)  # blocks for input dimension
        )
        
        # 9. Launch the dw backward kernel
        linear_backward_dw_kernel[dw_grid](
            x, dout, dw,
            batch_size, in_features, out_features,
            x.stride(0), x.stride(1),
            dout.stride(0), dout.stride(1),
            dw.stride(0), dw.stride(1),
        )
        
        # 10. Return gradients
        return dx, dw

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        """
        Triton implementation of a linear layer without bias (y = x @ W.t()).

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            device (str or torch.device): Device to store parameters.
            dtype (torch.dtype): Data type of parameters.

        Returns:
            None

        Explanation:
            1. Initialize parent class.
            2. Store feature dimensions.
            3. Initialize weight parameter with empty tensor.
            4. Initialize weights using Kaiming uniform.
        """
        # 1. Initialize parent class
        super(TritonLinear, self).__init__()
        
        # 2. Store feature dimensions
        self.in_features = in_features
        self.out_features = out_features
        
        # 3. Initialize weight parameter
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        
        # 4. Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialize parameters using Kaiming uniform initialization.

        Args:
            None

        Returns:
            None

        Explanation:
            1. Apply Kaiming uniform initialization to weights.
        """
        # 1. Apply Kaiming uniform initialization
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def get_parameter(self, name):
        """
        Get parameter by name.

        Args:
            name (str): Parameter name.

        Returns:
            torch.nn.Parameter: The requested parameter or None if not found.

        Explanation:
            1. Check if requested parameter matches 'weight'.
            2. Return weight parameter if match found, otherwise None.
        """
        # 1. Check parameter name
        if name == 'weight':
            # 2. Return matching parameter
            return self.weight
        return None
        
    def forward(self, x):
        """
        Forward pass using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor. [batch_size, in_features]

        Returns:
            torch.Tensor: Output tensor. [batch_size, out_features]

        Explanation:
            1. Apply TritonLinearFunction to input and weights.
            2. Return computed output.
        """
        # 1. Apply TritonLinearFunction
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

This looks good. The code implements a `TritonLinear` module that uses Triton kernels for both forward and backward passes, matching the functionality of `nn.Linear` with `bias=False`. I've updated all the docstrings to match the required format, with detailed type hints, shapes for tensors, and numbered explanations of steps.

Let's verify that we covered everything required:

1. ✅ Created a `TritonLinear` module that implements `nn.Linear` with `bias=False`
2. ✅ Implemented forward and backward passes using Triton kernels
3. ✅ Used autotune for block sizes in both forward and backward kernels
4. ✅ Implemented super blocks with corresponding GROUP_SIZE parameters
5. ✅ Added correct docstrings for all functions as required
6. ✅ Included the test code for correctness and performance

Everything looks good, and the code should run as expected. The final code has complete implementations of all required components.

# text
<think>
# Implementing a Triton Linear Layer with Forward and Backward Pass

I need to create a PyTorch module called `TritonLinear` that uses Triton kernels to implement the functionality of `nn.Linear`. I'll implement both forward and backward passes, and use autotuning to optimize performance.

## Understanding the Requirements

1. The implementation should be equivalent to `nn.Linear` but without bias
2. Need to implement:
   - Forward kernel: computes `y = x @ weight.t()`
   - Backward kernels: 
     - For input gradients: `dx = dout @ weight`
     - For weight gradients: `dw = dout.t() @ x`
3. Use autotune for block sizes
4. Implement super blocks to handle larger dimensions efficiently
5. Provide comprehensive docstrings that explain each component
6. Test correctness and performance using the benchmark code

## Approach

1. Define Triton kernels for forward and backward passes
2. Create an autograd function for proper gradient calculation
3. Create a PyTorch module that uses the autograd function
4. Add autotuning configurations for all kernels
5. Implement block-based computation with super blocks for performance
6. Test using the provided benchmark code

Let me implement this step by step.
</think>

```python
# filename: linear_triton.py
import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 1,
        }, num_stages=3, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_O': 16,
            'GROUP_SIZE_B': 1,
        }, num_stages=1, num_warps=2),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_forward_kernel(
    x_ptr, w_ptr, output_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_wo, stride_wi,
    stride_ob, stride_oi,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the forward pass of a linear layer: y = x @ w.t()

    Args:
        x_ptr (ptr): Pointer to the input tensor. [B, I]
        w_ptr (ptr): Pointer to the weight tensor. [O, I]
        output_ptr (ptr): Pointer to the output tensor. [B, O]
        B (int): Batch size.
        I (int): Input dimension.
        O (int): Output dimension.
        stride_xb (int): Stride for batch dimension of x.
        stride_xi (int): Stride for input dimension of x.
        stride_wo (int): Stride for output dimension of w.
        stride_wi (int): Stride for input dimension of w.
        stride_ob (int): Stride for batch dimension of output.
        stride_oi (int): Stride for output dimension of output.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension.

    Returns:
        None: Results are stored in output_ptr.

    Explanation:
        1. Map program IDs to blocks of the matrix multiplication.
        2. Initialize offsets and masks for boundary checking.
        3. Process multiple batch blocks using GROUP_SIZE_B.
        4. For each batch block, initialize accumulator to zeros.
        5. Iterate through input dimension in blocks of BLOCK_SIZE_I.
        6. Load blocks of input and weight matrices.
        7. Perform matrix multiplication using tl.dot.
        8. Store accumulated results to output tensor.
    """
    # 1. Map program IDs to blocks of the matrix multiplication
    pid_b = tl.program_id(axis=0)
    pid_o = tl.program_id(axis=1)
    
    # 2. Initialize offsets and masks
    offs_b_start = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_b = offs_b_start + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    mask_o = offs_o < O
    
    # 3. Process multiple batch blocks using GROUP_SIZE_B
    for group_idx in range(GROUP_SIZE_B):
        # 4. Initialize accumulator to zeros
        acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
        current_offs_b = offs_b[group_idx * BLOCK_SIZE_B:(group_idx + 1) * BLOCK_SIZE_B]
        mask_b = current_offs_b < B
        
        # 5. Iterate through input dimension in blocks
        for i in range(0, I, BLOCK_SIZE_I):
            mask_i = (i + offs_i) < I
            
            # 6. Load blocks of input and weight matrices
            x_ptrs = x_ptr + (current_offs_b[:, None] * stride_xb + (i + offs_i[None, :]) * stride_xi)
            x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_i[None, :], other=0.0)
            
            w_ptrs = w_ptr + (offs_o[None, :] * stride_wo + (i + offs_i[:, None]) * stride_wi)
            w = tl.load(w_ptrs, mask=mask_i[:, None] & mask_o[None, :], other=0.0)
            
            # 7. Perform matrix multiplication
            acc += tl.dot(x, w)
        
        # 8. Store results to output tensor
        output_ptrs = output_ptr + (current_offs_b[:, None] * stride_ob + offs_o[None, :] * stride_oi)
        tl.store(output_ptrs, acc, mask=mask_b[:, None] & mask_o[None, :])

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 1,
        }, num_stages=3, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 32,
            'GROUP_SIZE_B': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_O': 64,
            'GROUP_SIZE_B': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_B': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_O': 16,
            'GROUP_SIZE_B': 1,
        }, num_stages=1, num_warps=2),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_dx_kernel(
    dout_ptr, w_ptr, dx_ptr,
    B, O, I,
    stride_doutb, stride_douto,
    stride_wo, stride_wi,
    stride_dxb, stride_dxi,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the backward pass for the input gradient: dx = dout @ w

    Args:
        dout_ptr (ptr): Pointer to the output gradient tensor. [B, O]
        w_ptr (ptr): Pointer to the weight tensor. [O, I]
        dx_ptr (ptr): Pointer to the input gradient tensor. [B, I]
        B (int): Batch size.
        O (int): Output dimension.
        I (int): Input dimension.
        stride_doutb (int): Stride for batch dimension of dout.
        stride_douto (int): Stride for output dimension of dout.
        stride_wo (int): Stride for output dimension of w.
        stride_wi (int): Stride for input dimension of w.
        stride_dxb (int): Stride for batch dimension of dx.
        stride_dxi (int): Stride for input dimension of dx.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension.

    Returns:
        None: Results are stored in dx_ptr.

    Explanation:
        1. Map program IDs to blocks of the matrix multiplication.
        2. Initialize offsets and masks for boundary checking.
        3. Process multiple batch blocks using GROUP_SIZE_B.
        4. For each batch block, initialize accumulator to zeros.
        5. Iterate through output dimension in blocks of BLOCK_SIZE_O.
        6. Load blocks of dout and weight matrices.
        7. Perform matrix multiplication using tl.dot to compute input gradients.
        8. Store accumulated results to dx tensor.
    """
    # 1. Map program IDs to blocks
    pid_b = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)
    
    # 2. Initialize offsets and masks
    offs_b_start = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = offs_b_start + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    mask_i = offs_i < I
    
    # 3. Process multiple batch blocks
    for group_idx in range(GROUP_SIZE_B):
        # 4. Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
        current_offs_b = offs_b[group_idx * BLOCK_SIZE_B:(group_idx + 1) * BLOCK_SIZE_B]
        mask_b = current_offs_b < B
        
        # 5. Iterate through output dimension
        for o in range(0, O, BLOCK_SIZE_O):
            mask_o = (o + offs_o) < O
            
            # 6. Load blocks of dout and weight
            dout_ptrs = dout_ptr + (current_offs_b[:, None] * stride_doutb + (o + offs_o[None, :]) * stride_douto)
            dout = tl.load(dout_ptrs, mask=mask_b[:, None] & mask_o[None, :], other=0.0)
            
            w_ptrs = w_ptr + ((o + offs_o[:, None]) * stride_wo + offs_i[None, :] * stride_wi)
            w = tl.load(w_ptrs, mask=mask_o[:, None] & mask_i[None, :], other=0.0)
            
            # 7. Compute input gradients
            acc += tl.dot(dout, w)
        
        # 8. Store results
        dx_ptrs = dx_ptr + (current_offs_b[:, None] * stride_dxb + offs_i[None, :] * stride_dxi)
        tl.store(dx_ptrs, acc, mask=mask_b[:, None] & mask_i[None, :])

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 32,
            'GROUP_SIZE_O': 1,
        }, num_stages=3, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 64,
            'GROUP_SIZE_O': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 32,
            'GROUP_SIZE_O': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 64,
            'BLOCK_SIZE_I': 32,
            'BLOCK_SIZE_B': 64,
            'GROUP_SIZE_O': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_B': 32,
            'GROUP_SIZE_O': 4,
        }, num_stages=4, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 32,
            'BLOCK_SIZE_I': 64,
            'BLOCK_SIZE_B': 64,
            'GROUP_SIZE_O': 8,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_O': 16,
            'BLOCK_SIZE_I': 16,
            'BLOCK_SIZE_B': 16,
            'GROUP_SIZE_O': 1,
        }, num_stages=1, num_warps=2),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_dw_kernel(
    x_ptr, dout_ptr, dw_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_doutb, stride_douto,
    stride_dwo, stride_dwi,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute the backward pass for the weight gradient: dw = dout.t() @ x

    Args:
        x_ptr (ptr): Pointer to the input tensor. [B, I]
        dout_ptr (ptr): Pointer to the output gradient tensor. [B, O]
        dw_ptr (ptr): Pointer to the weight gradient tensor. [O, I]
        B (int): Batch size.
        I (int): Input dimension.
        O (int): Output dimension.
        stride_xb (int): Stride for batch dimension of x.
        stride_xi (int): Stride for input dimension of x.
        stride_doutb (int): Stride for batch dimension of dout.
        stride_douto (int): Stride for output dimension of dout.
        stride_dwo (int): Stride for output dimension of dw.
        stride_dwi (int): Stride for input dimension of dw.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        GROUP_SIZE_O (tl.constexpr): Group size for output dimension.

    Returns:
        None: Results are stored in dw_ptr.

    Explanation:
        1. Map program IDs to blocks of the matrix multiplication.
        2. Initialize offsets and masks for boundary checking.
        3. Process multiple output blocks using GROUP_SIZE_O.
        4. For each output block, initialize accumulator to zeros.
        5. Iterate through batch dimension in blocks of BLOCK_SIZE_B.
        6. Load blocks of x and dout matrices.
        7. Perform matrix multiplication using tl.dot to compute weight gradients.
        8. Store accumulated results to dw tensor.
    """
    # 1. Map program IDs to blocks
    pid_o = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)
    
    # 2. Initialize offsets and masks
    offs_o_start = pid_o * BLOCK_SIZE_O * GROUP_SIZE_O
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = offs_o_start + tl.arange(0, BLOCK_SIZE_O * GROUP_SIZE_O)
    offs_b = tl.arange(0, BLOCK_SIZE_B)
    mask_i = offs_i < I
    
    # 3. Process multiple output blocks
    for group_idx in range(GROUP_SIZE_O):
        # 4. Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
        current_offs_o = offs_o[group_idx * BLOCK_SIZE_O:(group_idx + 1) * BLOCK_SIZE_O]
        mask_o = current_offs_o < O
        
        # 5. Iterate through batch dimension
        for b in range(0, B, BLOCK_SIZE_B):
            mask_b = (b + offs_b) < B
            
            # 6. Load blocks of x and dout
            x_ptrs = x_ptr + ((b + offs_b)[:, None] * stride_xb + offs_i[None, :] * stride_xi)
            x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_i[None, :], other=0.0)
            
            dout_ptrs = dout_ptr + ((b + offs_b)[None, :] * stride_doutb + current_offs_o[:, None] * stride_douto)
            dout = tl.load(dout_ptrs, mask=mask_o[:, None] & mask_b[None, :], other=0.0)
            
            # 7. Compute weight gradients
            acc += tl.dot(dout, x)
        
        # 8. Store results
        dw_ptrs = dw_ptr + (current_offs_o[:, None] * stride_dwo + offs_i[None, :] * stride_dwi)
        tl.store(dw_ptrs, acc, mask=mask_o[:, None] & mask_i[None, :])

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """
        Forward pass for the linear layer.

        Args:
            ctx (torch.autograd.function.Context): Autograd context object.
            x (torch.Tensor): Input tensor. [batch_size, in_features]
            weight (torch.Tensor): Weight tensor. [out_features, in_features]

        Returns:
            torch.Tensor: Output tensor. [batch_size, out_features]

        Explanation:
            1. Extract dimensions from input tensors.
            2. Ensure inputs are contiguous in memory.
            3. Save inputs for backward pass.
            4. Create output tensor.
            5. Calculate grid dimensions for kernel launch.
            6. Launch the forward Triton kernel.
            7. Return the computed output.
        """
        # 1. Extract dimensions
        batch_size = x.shape[0]
        out_features = weight.shape[0]
        in_features = weight.shape[1]
        
        # 2. Ensure contiguous inputs
        x = x.contiguous()
        weight = weight.contiguous()
        
        # 3. Save inputs for backward
        ctx.save_for_backward(x, weight)
        
        # 4. Create output tensor
        output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
        
        # 5. Calculate grid dimensions
        grid = (
            triton.cdiv(batch_size, 128),  # blocks for batch dimension
            triton.cdiv(out_features, 128)  # blocks for output dimension
        )
        
        # 6. Launch the forward kernel
        linear_forward_kernel[grid](
            x, weight, output,
            batch_size, in_features, out_features,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        
        # 7. Return output
        return output
    
    @staticmethod
    def backward(ctx, dout):
        """
        Backward pass for the linear layer.

        Args:
            ctx (torch.autograd.function.Context): Autograd context object.
            dout (torch.Tensor): Gradient tensor from upstream. [batch_size, out_features]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - dx: Gradient with respect to input. [batch_size, in_features]
                - dw: Gradient with respect to weight. [out_features, in_features]

        Explanation:
            1. Retrieve saved tensors from context.
            2. Extract dimensions from tensors.
            3. Ensure gradients are contiguous in memory.
            4. Compute gradient with respect to input (dx).
            5. Calculate grid dimensions for dx computation.
            6. Launch the dx backward kernel.
            7. Compute gradient with respect to weights (dw).
            8. Calculate grid dimensions for dw computation.
            9. Launch the dw backward kernel.
            10. Return computed gradients.
        """
        # 1. Retrieve saved tensors
        x, weight = ctx.saved_tensors
        
        # 2. Extract dimensions
        batch_size = x.shape[0]
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        
        # 3. Ensure contiguous gradients
        dout = dout.contiguous()
        
        # 4. Prepare tensor for dx
        dx = torch.empty((batch_size, in_features), device=x.device, dtype=x.dtype)
        
        # 5. Calculate grid dimensions for dx
        dx_grid = (
            triton.cdiv(batch_size, 128),  # blocks for batch dimension
            triton.cdiv(in_features, 128)  # blocks for input dimension
        )
        
        # 6. Launch the dx backward kernel
        linear_backward_dx_kernel[dx_grid](
            dout, weight, dx,
            batch_size, out_features, in_features,
            dout.stride(0), dout.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
        )
        
        # 7. Prepare tensor for dw
        dw = torch.empty_like(weight)
        
        # 8. Calculate grid dimensions for dw
        dw_grid = (
            triton.cdiv(out_features, 128),  # blocks for output dimension
            triton.cdiv(in_features, 128)  # blocks for input dimension
        )
        
        # 9. Launch the dw backward kernel
        linear_backward_dw_kernel[dw_grid](
            x, dout, dw,
            batch_size, in_features, out_features,
            x.stride(0), x.stride(1),
            dout.stride(0), dout.stride(1),
            dw.stride(0), dw.stride(1),
        )
        
        # 10. Return gradients
        return dx, dw

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        """
        Triton implementation of a linear layer without bias (y = x @ W.t()).

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            device (str or torch.device): Device to store parameters.
            dtype (torch.dtype): Data type of parameters.

        Returns:
            None

        Explanation:
            1. Initialize parent class.
            2. Store feature dimensions.
            3. Initialize weight parameter with empty tensor.
            4. Initialize weights using Kaiming uniform.
        """
        # 1. Initialize parent class
        super(TritonLinear, self).__init__()
        
        # 2. Store feature dimensions
        self.in_features = in_features
        self.out_features = out_features
        
        # 3. Initialize weight parameter
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        
        # 4. Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialize parameters using Kaiming uniform initialization.

        Args:
            None

        Returns:
            None

        Explanation:
            1. Apply Kaiming uniform initialization to weights.
        """
        # 1. Apply Kaiming uniform initialization
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def get_parameter(self, name):
        """
        Get parameter by name.

        Args:
            name (str): Parameter name.

        Returns:
            torch.nn.Parameter: The requested parameter or None if not found.

        Explanation:
            1. Check if requested parameter matches 'weight'.
            2. Return weight parameter if match found, otherwise None.
        """
        # 1. Check parameter name
        if name == 'weight':
            # 2. Return matching parameter
            return self.weight
        return None
        
    def forward(self, x):
        """
        Forward pass using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor. [batch_size, in_features]

        Returns:
            torch.Tensor: Output tensor. [batch_size, out_features]

        Explanation:
            1. Apply TritonLinearFunction to input and weights.
            2. Return computed output.
        """
        # 1. Apply TritonLinearFunction
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
## Testing the Triton Linear Layer

To test the code:

1. Make sure you have PyTorch with CUDA support and Triton installed.
2. Save the above code as `linear_triton.py` in the same directory as `benchmark.py`.
3. Run the script to test correctness and performance compared to PyTorch's built-in linear layer.

```bash
python linear_triton.py
```

The script will:
1. Verify the correctness of the forward pass by comparing outputs with PyTorch's implementation
2. Benchmark performance of the forward pass
3. Verify the correctness of backward pass gradients
4. Benchmark performance of the backward pass

The results will be saved as JSON files and performance visualizations will be saved as PNG files in the current directory.
</test>

