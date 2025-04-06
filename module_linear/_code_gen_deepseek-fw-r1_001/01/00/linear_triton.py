# filename: linear_triton.py
import torch
import triton
import triton.language as tl
from torch.amp import custom_fwd, custom_bwd

class TritonLinear(torch.nn.Module):
    """
    A PyTorch module implementing a linear layer using Triton kernels for forward and backward passes.
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        device (str): Computation device ('cuda' or 'cpu').
        dtype (torch.dtype): Data type for parameters and computations.
    """
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, x, weight):
        """
        Forward pass of the Triton linear layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, I).
            weight (torch.Tensor): Weight tensor of shape (O, I).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, O).
        """
        ctx.save_for_backward(x, weight)
        B, I = x.shape
        O = weight.shape[0]
        output = torch.empty((B, O), device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(O, meta['BLOCK_SIZE_O']))
        forward_kernel[grid](
            x, weight, output,
            B, I, O,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        """
        Backward pass of the Triton linear layer.
        
        Args:
            grad_output (torch.Tensor): Gradient of output tensor of shape (B, O).
            
        Returns:
            Tuple[torch.Tensor]: Gradients for input and weight tensors.
        """
        x, weight = ctx.saved_tensors
        B, I = x.shape
        O = weight.shape[0]

        grad_input = torch.empty_like(x)
        grad_weight = torch.empty_like(weight)

        grid_dx = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
        backward_input_kernel[grid_dx](
            grad_output, weight, grad_input,
            B, O, I,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1),
        )

        grid_dw = lambda meta: (triton.cdiv(O, meta['BLOCK_SIZE_O']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
        backward_weight_kernel[grid_dw](
            x, grad_output, grad_weight,
            B, I, O,
            x.stride(0), x.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
        )

        return grad_input, grad_weight

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 128}, num_warps=16),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
):
    """Triton kernel for forward pass of linear layer."""
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, BLOCK_SIZE_I):
        i_offs = i + tl.arange(0, BLOCK_SIZE_I)
        mask_i = i_offs < I
        
        input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + i_offs[None, :] * stride_input_i
        weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + i_offs[None, :] * stride_weight_i
        
        mask_input = (offs_b[:, None] < B) & mask_i
        input = tl.load(input_ptrs, mask=mask_input, other=0.0)
        
        mask_weight = (offs_o[:, None] < O) & mask_i
        weight = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(input, tl.trans(weight), allow_tf32=True)
    
    mask_output = (offs_b[:, None] < B) & (offs_o[None, :] < O)
    tl.store(output_ptr + offs_b[:, None] * stride_output_b + offs_o[None, :] * stride_output_o,
             acc.to(tl.float16), mask=mask_output)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 128}, num_warps=16),
    ],
    key=['B', 'O', 'I']
)
@triton.jit
def backward_input_kernel(
    d_output_ptr, weight_ptr, grad_input_ptr,
    B, O, I,
    stride_dout_b, stride_dout_o,
    stride_weight_o, stride_weight_i,
    stride_grad_b, stride_grad_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
):
    """Triton kernel for input gradient computation."""
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for o in range(0, O, BLOCK_SIZE_O):
        o_offs = o + tl.arange(0, BLOCK_SIZE_O)
        mask_o = o_offs < O
        
        dout_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + o_offs[None, :] * stride_dout_o
        weight_ptrs = weight_ptr + o_offs[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
        
        mask_dout = (offs_b[:, None] < B) & mask_o
        dout = tl.load(dout_ptrs, mask=mask_dout, other=0.0)
        
        mask_weight = mask_o[:, None] & (offs_i[None, :] < I)
        weight = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(dout, weight, allow_tf32=True)
    
    mask_grad = (offs_b[:, None] < B) & (offs_i[None, :] < I)
    tl.store(grad_input_ptr + offs_b[:, None] * stride_grad_b + offs_i[None, :] * stride_grad_i,
             acc.to(tl.float16), mask=mask_grad)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_B': 128}, num_warps=16),
    ],
    key=['O', 'I', 'B']
)
@triton.jit
def backward_weight_kernel(
    input_ptr, d_output_ptr, grad_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_dout_b, stride_dout_o,
    stride_grad_o, stride_grad_i,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
):
    """Triton kernel for weight gradient computation."""
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    for b in range(0, B, BLOCK_SIZE_B):
        b_offs = b + tl.arange(0, BLOCK_SIZE_B)
        mask_b = b_offs < B
        
        input_ptrs = input_ptr + b_offs[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
        dout_ptrs = d_output_ptr + b_offs[:, None] * stride_dout_b + offs_o[None, :] * stride_dout_o
        
        mask_input = mask_b[:, None] & (offs_i[None, :] < I)
        input = tl.load(input_ptrs, mask=mask_input, other=0.0)
        
        mask_dout = mask_b[:, None] & (offs_o[None, :] < O)
        dout = tl.load(dout_ptrs, mask=mask_dout, other=0.0)
        
        acc += tl.dot(tl.trans(dout), input, allow_tf32=True)
    
    mask_grad = (offs_o[:, None] < O) & (offs_i[None, :] < I)
    tl.store(grad_weight_ptr + offs_o[:, None] * stride_grad_o + offs_i[None, :] * stride_grad_i,
             acc.to(tl.float16), mask=mask_grad)

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
