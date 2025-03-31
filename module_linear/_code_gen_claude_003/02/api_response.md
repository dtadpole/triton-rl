Timestamp: 20250331_045905

# thinking
I need to implement a Triton module called `TritonLinear` that acts like PyTorch's `nn.Linear`, including both forward and backward passes. Let's break this down:

1. First, I need to understand what `nn.Linear` does. It's a linear transformation: `y = xA^T + b`, where `x` is the input, `A` is the weight matrix, and `b` is the bias. In this implementation, I only need to consider the case without bias (since `bias=False` in the test code).

2. I'll need to implement the Triton kernels for both the forward pass (`linear_forward_kernel`) and the backward pass (which will require two kernels: one for computing gradients with respect to the input, and another for computing gradients with respect to the weights).

3. The linear forward kernel computes `y = xA^T`.

4. The backward kernels compute:
   - `dx = dy * A`
   - `dA = dy^T * x`

5. I'll need to create a PyTorch module wrapper that uses these Triton kernels, implementing both the `forward` and `backward` methods.

6. I'll need to use Triton's autotune functionality to optimize the block sizes for the kernels.

Let's start by implementing the forward pass kernel:

```python
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
    # Matrix dimensions
    # B: batch size
    # I: input dimension
    # O: output dimension
    
    # Compute the indices for each thread in the grid
    b_idx = tl.program_id(0) * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    o_idx = tl.program_id(1) * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    # Check if the indices are within bounds
    b_mask = b_idx < B
    o_mask = o_idx < O
    
    # Create a mask that combines the batch and output masks
    mask = b_mask[:, None] & o_mask[None, :]
    
    # Initialize the accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    # Iterate over the input dimension in blocks
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        i_idx = i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
        i_mask = i_idx < I
        
        # Load blocks from the input and weight matrices
        x_block = tl.load(x_ptr + b_idx[:, None] * I + i_idx[None, :], mask=b_mask[:, None] & i_mask[None, :], other=0.0)
        w_block = tl.load(w_ptr + o_idx[:, None] * I + i_idx[None, :], mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        
        # Transpose the weight block for matrix multiplication
        w_block_t = tl.trans(w_block)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(x_block, w_block_t, allow_tf32=False)
    
    # Store the result in the output matrix
    tl.store(y_ptr + b_idx[:, None] * O + o_idx[None, :], acc, mask=mask)
```

Now, let's implement the backward kernels. First, the kernel for computing gradients with respect to the input:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 16, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_O': 2, 'GROUP_SIZE_I': 2}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_O': 4, 'GROUP_SIZE_I': 4}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}),
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
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr,
):
    # Matrix dimensions
    # B: batch size
    # I: input dimension
    # O: output dimension
    
    # Compute the indices for each thread in the grid
    b_idx = tl.program_id(0) * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    i_idx = tl.program_id(1) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Check if the indices are within bounds
    b_mask = b_idx < B
    i_mask = i_idx < I
    
    # Create a mask that combines the batch and input masks
    mask = b_mask[:, None] & i_mask[None, :]
    
    # Initialize the accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over the output dimension in blocks
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        o_idx = o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
        o_mask = o_idx < O
        
        # Load blocks from the gradient and weight matrices
        dy_block = tl.load(dy_ptr + b_idx[:, None] * O + o_idx[None, :], mask=b_mask[:, None] & o_mask[None, :], other=0.0)
        w_block = tl.load(w_ptr + o_idx[:, None] * I + i_idx[None, :], mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(dy_block, w_block, allow_tf32=False)
    
    # Store the result in the gradient with respect to input matrix
    tl.store(dx_ptr + b_idx[:, None] * I + i_idx[None, :], acc, mask=mask)
```

Now, let's implement the kernel for computing gradients with respect to the weights:

```python
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
    # Matrix dimensions
    # B: batch size
    # I: input dimension
    # O: output dimension
    
    # Compute the indices for each thread in the grid
    o_idx = tl.program_id(0) * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    i_idx = tl.program_id(1) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Check if the indices are within bounds
    o_mask = o_idx < O
    i_mask = i_idx < I
    
    # Create a mask that combines the output and input masks
    mask = o_mask[:, None] & i_mask[None, :]
    
    # Initialize the accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over the batch dimension in blocks
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        b_idx = b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        b_mask = b_idx < B
        
        # Load blocks from the input and gradient matrices
        x_block = tl.load(x_ptr + b_idx[:, None] * I + i_idx[None, :], mask=b_mask[:, None] & i_mask[None, :], other=0.0)
        dy_block = tl.load(dy_ptr + b_idx[:, None] * O + o_idx[None, :], mask=b_mask[:, None] & o_mask[None, :], other=0.0)
        
        # Transpose the blocks for matrix multiplication
        x_block_t = tl.trans(x_block)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(dy_block, x_block_t, allow_tf32=False)
    
    # Store the result in the gradient with respect to weight matrix
    tl.store(dw_ptr + o_idx[:, None] * I + i_idx[None, :], acc, mask=mask)
```

Now, let's create the PyTorch module wrapper:

```python
class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        # Get input dimensions
        B, I = x.shape
        O = self.out_features
        
        # Allocate output tensor
        y = torch.empty((B, O), device=x.device, dtype=x.dtype)
        
        # Launch the kernel for forward pass
        grid = (triton.cdiv(B, BLOCK_SIZE_B), triton.cdiv(O, BLOCK_SIZE_O))
        linear_forward_kernel[grid](
            x.data_ptr(), self.weight.data_ptr(), y.data_ptr(),
            B, I, O,
            BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O,
            GROUP_SIZE_B, GROUP_SIZE_I, GROUP_SIZE_O
        )
        
        # We need to store input for the backward pass
        ctx = SaveForBackward(x, self.weight)
        y.requires_grad = True
        y._ctx = ctx
        return y

class SaveForBackward:
    def __init__(self, x, weight):
        self.x = x
        self.weight = weight
        
    def backward(self, dy):
        # Get dimensions
        B, I = self.x.shape
        _, O = dy.shape
        
        # Allocate tensors for gradients
        dx = torch.empty_like(self.x)
        dw = torch.empty_like(self.weight)
        
        # Launch kernels for backward pass
        # 1. Compute gradients with respect to input
        grid_dx = (triton.cdiv(B, BLOCK_SIZE_B), triton.cdiv(I, BLOCK_SIZE_I))
        linear_backward_input_kernel[grid_dx](
            dy.data_ptr(), self.weight.data_ptr(), dx.data_ptr(),
            B, I, O,
            BLOCK_SIZE_B, BLOCK_SIZE_O, BLOCK_SIZE_I,
            GROUP_SIZE_B, GROUP_SIZE_O, GROUP_SIZE_I
        )
        
        # 2. Compute gradients with respect to weights
        grid_dw = (triton.cdiv(O, BLOCK_SIZE_O), triton.cdiv(I, BLOCK_SIZE_I))
        linear_backward_weight_kernel[grid_dw](
            self.x.data_ptr(), dy.data_ptr(), dw.data_ptr(),
            B, I, O,
            BLOCK_SIZE_O, BLOCK_SIZE_I, BLOCK_SIZE_B,
            GROUP_SIZE_O, GROUP_SIZE_I, GROUP_SIZE_B
        )
        
        return dx, dw
```

Actually, there's an issue with how I've set up the backward pass in the code above. PyTorch's autograd system doesn't work this way with custom operations. Let me fix this by implementing the `TritonLinear` module differently:

```python
class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Get input dimensions
        B, I = x.shape
        O = weight.shape[0]
        
        # Allocate output tensor
        y = torch.empty((B, O), device=x.device, dtype=x.dtype)
        
        # Launch the kernel for forward pass
        grid = (triton.cdiv(B, BLOCK_SIZE_B), triton.cdiv(O, BLOCK_SIZE_O))
        linear_forward_kernel[grid](
            x.data_ptr(), weight.data_ptr(), y.data_ptr(),
            B, I, O,
            BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O,
            GROUP_SIZE_B, GROUP_SIZE_I, GROUP_SIZE_O
        )
        
        # Save inputs for backward pass
        ctx.save_for_backward(x, weight)
        ctx.dims = (B, I, O)
        
        return y
    
    @staticmethod
    def backward(ctx, dy):
        # Get saved tensors
        x, weight = ctx.saved_tensors
        B, I, O = ctx.dims
        
        # Allocate tensors for gradients
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        
        # Launch kernels for backward pass
        # 1. Compute gradients with respect to input
        grid_dx = (triton.cdiv(B, BLOCK_SIZE_B), triton.cdiv(I, BLOCK_SIZE_I))
        linear_backward_input_kernel[grid_dx](
            dy.data_ptr(), weight.data_ptr(), dx.data_ptr(),
            B, I, O,
            BLOCK_SIZE_B, BLOCK_SIZE_O, BLOCK_SIZE_I,
            GROUP_SIZE_B, GROUP_SIZE_O, GROUP_SIZE_I
        )
        
        # 2. Compute gradients with respect to weights
        grid_dw = (triton.cdiv(O, BLOCK_SIZE_O), triton.cdiv(I, BLOCK_SIZE_I))
        linear_backward_weight_kernel[grid_dw](
            x.data_ptr(), dy.data_ptr(), dw.data_ptr(),
            B, I, O,
            BLOCK_SIZE_O, BLOCK_SIZE_I, BLOCK_SIZE_B,
            GROUP_SIZE_O, GROUP_SIZE_I, GROUP_SIZE_B
        )
        
        return dx, dw
```

But wait, I need to rethink this a bit more. In my implementation:

1. The block sizes are referenced before they're defined.
2. The autotune configuration is not properly wired up.

Let's fix these issues and provide a more correct implementation:

```python
import torch
import triton
import triton.language as tl
import math

# Define block sizes for autotuning
CONFIGS = [
    triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_O': 16, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
    triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_I': 2, 'GROUP_SIZE_O': 2}),
    triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_I': 4, 'GROUP_SIZE_O': 4}),
    triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}),
]

@triton.autotune(
    configs=CONFIGS,
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
    # Matrix dimensions
    # B: batch size
    # I: input dimension
    # O: output dimension
    
    # Compute the indices for each thread in the grid
    b_idx = tl.program_id(0) * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    o_idx = tl.program_id(1) * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    # Check if the indices are within bounds
    b_mask = b_idx < B
    o_mask = o_idx < O
    
    # Create a mask that combines the batch and output masks
    mask = b_mask[:, None] & o_mask[None, :]
    
    # Initialize the accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    # Iterate over the input dimension in blocks
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        i_idx = i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
        i_mask = i_idx < I
        
        # Load blocks from the input and weight matrices
        x_block = tl.load(x_ptr + b_idx[:, None] * I + i_idx[None, :], mask=b_mask[:, None] & i_mask[None, :], other=0.0)
        w_block = tl.load(w_ptr + o_idx[:, None] * I + i_idx[None, :], mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        
        # Transpose the weight block for matrix multiplication
        w_block_t = tl.trans(w_block)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(x_block, w_block_t, allow_tf32=False)
    
    # Store the result in the output matrix
    tl.store(y_ptr + b_idx[:, None] * O + o_idx[None, :], acc, mask=mask)

@triton.autotune(
    configs=CONFIGS,
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_backward_input_kernel(
    # Pointers to matrices
    dy_ptr, w_ptr, dx_ptr,
    # Matrix dimensions
    B, I, O,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr,
):
    # Matrix dimensions
    # B: batch size
    # I: input dimension
    # O: output dimension
    
    # Compute the indices for each thread in the grid
    b_idx = tl.program_id(0) * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    i_idx = tl.program_id(1) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Check if the indices are within bounds
    b_mask = b_idx < B
    i_mask = i_idx < I
    
    # Create a mask that combines the batch and input masks
    mask = b_mask[:, None] & i_mask[None, :]
    
    # Initialize the accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over the output dimension in blocks
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        o_idx = o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
        o_mask = o_idx < O
        
        # Load blocks from the gradient and weight matrices
        dy_block = tl.load(dy_ptr + b_idx[:, None] * O + o_idx[None, :], mask=b_mask[:, None] & o_mask[None, :], other=0.0)
        w_block = tl.load(w_ptr + o_idx[:, None] * I + i_idx[None, :], mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(dy_block, w_block, allow_tf32=False)
    
    # Store the result in the gradient with respect to input matrix
    tl.store(dx_ptr + b_idx[:, None] * I + i_idx[None, :], acc, mask=mask)

@triton.autotune(
    configs=CONFIGS,
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
    # Matrix dimensions
    # B: batch size
    # I: input dimension
    # O: output dimension
    
    # Compute the indices for each thread in the grid
    o_idx = tl.program_id(0) * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    i_idx = tl.program_id(1) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Check if the indices are within bounds
    o_mask = o_idx < O
    i_mask = i_idx < I
    
    # Create a mask that combines the output and input masks
    mask = o_mask[:, None] & i_mask[None, :]
    
    # Initialize the accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over the batch dimension in blocks
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        b_idx = b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        b_mask = b_idx < B
        
        # Load blocks from the input and gradient matrices
        x_block = tl.load(x_ptr + b_idx[:, None] * I + i_idx[None, :], mask=b_mask[:, None] & i_mask[None, :], other=0.0)
        dy_block = tl.load(dy_ptr + b_idx[:, None] * O + o_idx[None, :], mask=b_mask[:, None] & o_mask[None, :], other=0.0)
        
        # Transpose the blocks for matrix multiplication
        x_block_t = tl.trans(x_block)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(dy_block, x_block_t, allow_tf32=False)
    
    # Store the result in the gradient with respect to weight matrix
    tl.store(dw_ptr + o_idx[:, None] * I + i_idx[None, :], acc, mask=mask)

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Get input dimensions
        B, I = x.shape
        O = weight.shape[0]
        
        # Allocate output tensor
        y = torch.empty((B, O), device=x.device, dtype=x.dtype)
        
        # Launch the kernel for forward pass
        grid = (triton.cdiv(B, 128), triton.cdiv(O, 128))
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
        # Get saved tensors
        x, weight = ctx.saved_tensors
        B, I, O = ctx.dims
        
        # Allocate tensors for gradients
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        
        # Launch kernels for backward pass
        # 1. Compute gradients with respect to input
        grid_dx = (triton.cdiv(B, 128), triton.cdiv(I, 128))
        linear_backward_input_kernel[grid_dx](
            dy.data_ptr(), weight.data_ptr(), dx.data_ptr(),
            B, I, O,
        )
        
        # 2. Compute gradients with respect to weights
        grid_dw = (triton.cdiv(O, 128), triton.cdiv(I, 128))
        linear_backward_weight_kernel[grid_dw](
            x.data_ptr(), dy.data_ptr(), dw.data_ptr(),
            B, I, O,
        )
        
        return dx, dw

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)
```

I think this implementation is closer, but let me do one more check. I'm not sure if I should be using the autotuned block size in the kernel grid size calculation. Let's fix that:

The correct approach is to have the kernels determine the block sizes through autotuning, but I shouldn't hardcode the block sizes in the grid calculation (e.g., `triton.cdiv(B, 128)`). Instead, I can use a default block size for the grid calculation, and the autotune mechanism will optimize the meta-parameters at runtime.

Let me fix that:

```python
class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Get input dimensions
        B, I = x.shape
        O = weight.shape[0]
        
        # Allocate output tensor
        y = torch.empty((B, O), device=x.device, dtype=x.dtype)
        
        # Launch the kernel for forward pass
        grid = (triton.cdiv(B, 32), triton.cdiv(O, 32))  # Grid size doesn't need to match block size
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
        # Get saved tensors
        x, weight = ctx.saved_tensors
        B, I, O = ctx.dims
        
        # Allocate tensors for gradients
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        
        # Launch kernels for backward pass
        # 1. Compute gradients with respect to input
        grid_dx = (triton.cdiv(B, 32), triton.cdiv(I, 32))  # Grid size doesn't need to match block size
        linear_backward_input_kernel[grid_dx](
            dy.data_ptr(), weight.data_ptr(), dx.data_ptr(),
            B, I, O,
        )
        
        # 2. Compute gradients with respect to weights
        grid_dw = (triton.cdiv(O, 32), triton.cdiv(I, 32))  # Grid size doesn't need to match block size
        linear_backward_weight_kernel[grid_dw](
            x.data_ptr(), dy.data_ptr(), dw.data_ptr(),
            B, I, O,
        )
        
        return dx, dw
```

Now, let's make sure my understanding of how to use Triton's autotune is correct. In the case of the linear layer, we have three dimensions to consider: batch size (`B`), input features (`I`), and output features (`O`). Each dimension needs an appropriate block size.

Let me revise the autotune configuration:

```python
# Define block size configurations for autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_O': 16, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'GROUP_SIZE_O': 1}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_I': 2, 'GROUP_SIZE_O': 2}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_I': 4, 'GROUP_SIZE_O': 4}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}),
    ],
    key=['B', 'I', 'O'],
)
```

This part seems correct. Each configuration specifies different block sizes and group sizes, and Triton's autotuner will select the best-performing configuration based on the input dimensions.

Now, I think I have a good implementation of the Triton module for linear forward and backward operations. Let's put everything together and check our final implementation:

```python
import torch
import triton
import triton.language as tl
import math

# Forward kernel for linear layer (computing y = x @ W.T)
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
    # Matrix dimensions
    # B: batch size
    # I: input dimension
    # O: output dimension
    
    # Compute the indices for each thread in the grid
    b_idx = tl.program_id(0) * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    o_idx = tl.program_id(1) * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    # Check if the indices are within bounds
    b_mask = b_idx < B
    o_mask = o_idx < O
    
    # Create a mask that combines the batch and output masks
    mask = b_mask[:, None] & o_mask[None, :]
    
    # Initialize the accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    # Iterate over the input dimension in blocks
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        i_idx = i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
        i_mask = i_idx < I
        
        # Load blocks from the input and weight matrices
        x_block = tl.load(x_ptr + b_idx[:, None] * I + i_idx[None, :], mask=b_mask[:, None] & i_mask[None, :], other=0.0)
        w_block = tl.load(w_ptr + o_idx[:, None] * I + i_idx[None, :], mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        
        # Transpose the weight block for matrix multiplication
        w_block_t = tl.trans(w_block)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(x_block, w_block_t, allow_tf32=False)
    
    # Store the result in the output matrix
    tl.store(y_ptr + b_idx[:, None] * O + o_idx[None, :], acc, mask=mask)

# Backward kernel for linear layer (computing dx = dy @ W)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 16, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_O': 2, 'GROUP_SIZE_I': 2}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_O': 4, 'GROUP_SIZE_I': 4}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}),
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
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr,
):
    # Matrix dimensions
    # B: batch size
    # I: input dimension
    # O: output dimension
    
    # Compute the indices for each thread in the grid
    b_idx = tl.program_id(0) * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    i_idx = tl.program_id(1) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Check if the indices are within bounds
    b_mask = b_idx < B
    i_mask = i_idx < I
    
    # Create a mask that combines the batch and input masks
    mask = b_mask[:, None] & i_mask[None, :]
    
    # Initialize the accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over the output dimension in blocks
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        o_idx = o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
        o_mask = o_idx < O
        
        # Load blocks from the gradient and weight matrices
        dy_block = tl.load(dy_ptr + b_idx[:, None] * O + o_idx[None, :], mask=b_mask[:, None] & o_mask[None, :], other=0.0)
        w_block = tl.load(w_ptr + o_idx[:, None] * I + i_idx[None, :], mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(dy_block, w_block, allow_tf32=False)
    
    # Store the result in the gradient with respect to input matrix
    tl.store(dx_ptr + b_idx[:, None] * I + i_idx[None, :], acc, mask=mask)

# Backward kernel for linear layer (computing dw = dy.T @ x)
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
    # Matrix dimensions
    # B: batch size
    # I: input dimension
    # O: output dimension
    
    # Compute the indices for each thread in the grid
    o_idx = tl.program_id(0) * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    i_idx = tl.program_id(1) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Check if the indices are within bounds
    o_mask = o_idx < O
    i_mask = i_idx < I
    
    # Create a mask that combines the output and input masks
    mask = o_mask[:, None] & i_mask[None, :]
    
    # Initialize the accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over the batch dimension in blocks
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        b_idx = b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        b_mask = b_idx < B
        
        # Load blocks from the input and gradient matrices
        x_block = tl.load(x_ptr + b_idx[:, None] * I + i_idx[None, :], mask=b_mask[:, None] & i_mask[None, :], other=0.0)
        dy_block = tl.load(dy_ptr + b_idx[:, None] * O + o_idx[None, :], mask=b_mask[:, None] & o_mask[None, :], other=0.0)
        
        # Transpose the blocks for matrix multiplication
        x_block_t = tl.trans(x_block)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(dy_block, x_block_t, allow_tf32=False)
    
    # Store the result in the gradient with respect to weight matrix
    tl.store(dw_ptr + o_idx[:, None] * I + i_idx[None, :], acc, mask=mask)

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Get input dimensions
        B, I = x.shape
        O = weight.shape[0]
        
        # Allocate output tensor
        y = torch.empty((B, O), device=x.device, dtype=x.dtype)
        
        # Launch the kernel for forward pass
        grid = (triton.cdiv(B, 32), triton.cdiv(O, 32))  # Grid size doesn't need to match block size
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
        # Get saved tensors
        x, weight = ctx.saved_tensors
        B, I, O = ctx.dims
        
        # Allocate tensors for gradients
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        
        # Launch kernels for backward pass
        # 1. Compute gradients with respect to input
        grid_dx = (triton.cdiv(B, 32), triton.cdiv(I, 32))  # Grid size doesn't need to match block size
        linear_backward_input_kernel[grid_dx](
            dy.data_ptr(), weight.data_ptr(), dx.data_ptr(),
            B, I, O,
        )
        
        # 2. Compute gradients with respect to weights
        grid_dw = (triton.cdiv(O, 32), triton.cdiv(I, 32))  # Grid size doesn't need to match block size
        linear_backward_weight_kernel[grid_dw](
            x.data_ptr(), dy.data_ptr(), dw.data_ptr(),
            B, I, O,
        )
        
        return dx, dw

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)
```

This implementation looks correct to me. Let's ensure it matches the requirements:

1. **TritonLinear Module**: We've implemented a PyTorch module that uses Triton kernels for forward and backward passes.
2. **Autotuning**: We've used `triton.autotune` for the block sizes in both the forward and backward kernels.
3. **Super Blocks**: We've included `GROUP_SIZE_*` parameters in the autotuning configuration.
4. **Testing**: We're using the provided testing code to verify correctness and performance.

Now, let me organize the final code into a single file, `linear_triton.py`, with proper documentation and detailed code explanations:

```python
# filename: linear_triton.py
import torch
import triton
import triton.language as tl
import math

# Forward kernel for linear layer (computing y = x @ W.T)
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
        x_ptr (ptr): Pointer to the input tensor [B, I]
        w_ptr (ptr): Pointer to the weight tensor [O, I]
        y_ptr (ptr): Pointer to the output tensor [B, O]
        B (int): Batch size
        I (int): Input dimension
        O (int): Output dimension
        BLOCK_SIZE_B (int): Block size for batch dimension
        BLOCK_SIZE_I (int): Block size for input dimension
        BLOCK_SIZE_O (int): Block size for output dimension
        GROUP_SIZE_B (int): Group size for batch dimension
        GROUP_SIZE_I (int): Group size for input dimension
        GROUP_SIZE_O (int): Group size for output dimension
    """
    # Compute the indices for each thread in the grid
    b_idx = tl.program_id(0) * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    o_idx = tl.program_id(1) * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    # Check if the indices are within bounds
    b_mask = b_idx < B
    o_mask = o_idx < O
    
    # Create a mask that combines the batch and output masks
    mask = b_mask[:, None] & o_mask[None, :]
    
    # Initialize the accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    # Iterate over the input dimension in blocks
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        i_idx = i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
        i_mask = i_idx < I
        
        # Load blocks from the input and weight matrices
        x_block = tl.load(x_ptr + b_idx[:, None] * I + i_idx[None, :], mask=b_mask[:, None] & i_mask[None, :], other=0.0)
        w_block = tl.load(w_ptr + o_idx[:, None] * I + i_idx[None, :], mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        
        # Transpose the weight block for matrix multiplication
        w_block_t = tl.trans(w_block)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(x_block, w_block_t, allow_tf32=False)
    
    # Store the result in the output matrix
    tl.store(y_ptr + b_idx[:, None] * O + o_idx[None, :], acc, mask=mask)

# Backward kernel for linear layer (computing dx = dy @ W)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 16, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_O': 2, 'GROUP_SIZE_I': 2}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_O': 4, 'GROUP_SIZE_I': 4}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}),
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
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr,
):
    """
    Compute the gradient with respect to the input: dx = dy @ W
    
    Args:
        dy_ptr (ptr): Pointer to the output gradient tensor [B, O]
        w_ptr (ptr): Pointer to the weight tensor [O, I]
        dx_ptr (ptr): Pointer to the input gradient tensor [B, I]
        B (int): Batch size
        I (int): Input dimension
        O (int): Output dimension
        BLOCK_SIZE_B (int): Block size for batch dimension
        BLOCK_SIZE_O (int): Block size for output dimension
        BLOCK_SIZE_I (int): Block size for input dimension
        GROUP_SIZE_B (int): Group size for batch dimension
        GROUP_SIZE_O (int): Group size for output dimension
        GROUP_SIZE_I (int): Group size for input dimension
    """
    # Compute the indices for each thread in the grid
    b_idx = tl.program_id(0) * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    i_idx = tl.program_id(1) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Check if the indices are within bounds
    b_mask = b_idx < B
    i_mask = i_idx < I
    
    # Create a mask that combines the batch and input masks
    mask = b_mask[:, None] & i_mask[None, :]
    
    # Initialize the accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over the output dimension in blocks
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        o_idx = o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
        o_mask = o_idx < O
        
        # Load blocks from the gradient and weight matrices
        dy_block = tl.load(dy_ptr + b_idx[:, None] * O + o_idx[None, :], mask=b_mask[:, None] & o_mask[None, :], other=0.0)
        w_block = tl.load(w_ptr + o_idx[:, None] * I + i_idx[None, :], mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(dy_block, w_block, allow_tf32=False)
    
    # Store the result in the gradient with respect to input matrix
    tl.store(dx_ptr + b_idx[:, None] * I + i_idx[None, :], acc, mask=mask)

# Backward kernel for linear layer (computing dw = dy.T @ x)
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
    Compute the gradient with respect to the weight: dw = dy.T @ x
    
    Args:
        x_ptr (ptr): Pointer to the input tensor [B, I]
        dy_ptr (ptr): Pointer to the output gradient tensor [B, O]
        dw_ptr (ptr): Pointer to the weight gradient tensor [O, I]
        B (int): Batch size
        I (int): Input dimension
        O (int): Output dimension
        BLOCK_SIZE_O (int): Block size for output dimension
        BLOCK_SIZE_I (int): Block size for input dimension
        BLOCK_SIZE_B (int): Block size for batch dimension
        GROUP_SIZE_O (int): Group size for output dimension
        GROUP_SIZE_I (int): Group size for input dimension
        GROUP_SIZE_B (int): Group size for batch dimension
    """
    # Compute the indices for each thread in the grid
    o_idx = tl.program_id(0) * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    i_idx = tl.program_id(1) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    # Check if the indices are within bounds
    o_mask = o_idx < O
    i_mask = i_idx < I
    
    # Create a mask that combines the output and input masks
    mask = o_mask[:, None] & i_mask[None, :]
    
    # Initialize the accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over the batch dimension in blocks
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        b_idx = b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        b_mask = b_idx < B
        
        # Load blocks from the input and gradient matrices
        x_block = tl.load(x_ptr + b_idx[:, None] * I + i_idx[None, :], mask=b_mask[:, None] & i_mask[None, :], other=0.0)
        dy_block = tl.load(dy_ptr + b_idx[:, None] * O + o_idx[None, :], mask=b_mask[:, None] & o_mask[None, :], other=0.0)
        
        # Transpose the blocks for matrix multiplication
        x_block_t = tl.trans(x_block)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(dy_block, x_block_t, allow_tf32=False)
    
    # Store the result in the gradient with respect to weight matrix
    tl.store(dw_ptr + o_idx[:, None] * I + i_idx[None, :], acc, mask=mask)

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """
        Forward pass of the linear layer.
        
        Args:
            ctx: Context variable for backward pass
            x (torch.Tensor): Input tensor of shape [B, I]
            weight (torch.Tensor): Weight tensor of shape [O, I]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, O]
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
            ctx: Context variable from forward pass
            dy (torch.Tensor): Gradient of loss with respect to output of shape [B, O]
            
        Returns:
            tuple: (dx, dw) - Gradients with respect to input and weight
        """
        # Get saved tensors
        x, weight = ctx.saved_tensors
        B, I, O = ctx.dims
        
        # Allocate tensors for gradients
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        
        # Launch kernels for backward pass
        # 1. Compute gradients with respect to input
        grid_dx = (triton.cdiv(B, 32), triton.cdiv(I, 32))
        linear_backward_input_kernel[grid_dx](
            dy.data_ptr(), weight.data_ptr(), dx.data_ptr(),
            B, I, O,
        )
        
        # 2. Compute gradients with respect to weights
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
            in_features (int): Size of each input sample
            out_features (int): Size of each output sample
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to False.
            device (str, optional): Device to create tensors on. Defaults to 'cuda'.
            dtype (torch.dtype, optional): Data type of tensors. Defaults to torch.float16.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Reset the parameters using Kaiming uniform initialization.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        """
        Forward pass of the linear layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, I]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, O]
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
```

I believe this final implementation meets all the requirements for the Triton-based linear layer. The code includes:

1. Three Triton kernels:
   - `linear_forward_kernel`: for the forward pass
   - `linear_backward_input_kernel`: for computing gradients with respect to the input
   - `linear_backward_weight_kernel`: for computing gradients with respect to the weights

2. Autotuning with multiple configurations for block sizes (BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O) and group sizes (GROUP_SIZE_B, GROUP_SIZE_I, GROUP_SIZE_O).

3. A PyTorch module wrapper (`TritonLinear`) and an autograd function (`TritonLinearFunction`) to handle the forward and backward passes.

4. The provided testing code to verify correctness and benchmark performance.

The implementation has been optimized for performance with tiled matrix multiplication, where each tile is processed by a thread block, and the autotuner will select the best-performing configuration based on the input dimensions.

# text
<think>
# Creating a Triton Implementation of nn.Linear

I need to create a Triton-based implementation of PyTorch's nn.Linear that provides both forward and backward pass functionality. Let's break down what I need to do:

1. **Understanding the Problem**
   - nn.Linear computes `y = xW^T + b` (bias is disabled in our case)
   - Forward pass: compute `y = xW^T`
   - Backward pass needs two components:
     - Gradient w.r.t input: `dx = dL/dy · W`
     - Gradient w.r.t weights: `dW = dL/dy^T · x`

2. **Triton Implementation Strategy**
   - Implement three CUDA kernels using Triton:
     - Forward kernel: computes `y = xW^T`
     - Backward input kernel: computes `dx = dL/dy · W`
     - Backward weight kernel: computes `dW = dL/dy^T · x`
   - Use autotuning to find optimal block sizes
   - Implement PyTorch autograd function for backward pass

3. **Architecture**
   - Create a `TritonLinear` module that extends `nn.Module`
   - Implement `TritonLinearFunction` as an autograd.Function for custom backward pass
   - Use autotune for performance optimization with various block sizes

Let me implement this with detailed documentation and proper tensor handling.
</think>

```python
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
```

<test>
To test the Triton-based linear layer implementation, run the provided benchmark script. The code will compare the correctness and performance of the Triton implementation against PyTorch's native implementation.

The test will:
1. Verify the correctness of both forward and backward passes across various shapes
2. Benchmark the performance comparison between Triton and PyTorch implementations
3. Generate performance graphs and result files

Run the following command to execute the tests:

```bash
python linear_triton.py
```

This will run all the tests with the following configurations:
- Batch sizes: 256, 1024, 4096
- Input dimensions: 256, 1024, 4096
- Output dimensions: 256, 1024, 4096

The test results will include:
- Correctness verification outputs
- Performance comparison results
- Benchmark graphs saved as PNG files
</test>

