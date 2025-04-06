# filename: linear_triton.py

import torch
import triton
import triton.language as tl

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        """
        Implements a linear layer using Triton kernels with autotuned block sizes.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            device (str): Computation device (default: 'cuda').
            dtype (torch.dtype): Data type for parameters (default: torch.float16).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(in_features, out_features, device=device, dtype=dtype))
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return _TritonLinearFunction.apply(x, self.weight)

class _TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """
        Compute forward pass using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (B, I).
            weight (torch.Tensor): Weight tensor of shape (I, O).

        Returns:
            torch.Tensor: Output tensor of shape (B, O).
        """
        ctx.save_for_backward(x, weight)
        B, I = x.shape
        O = weight.shape[1]
        output = torch.empty((B, O), device=x.device, dtype=x.dtype)

        def grid(meta):
            return (triton.cdiv(B, meta['BLOCK_SIZE_B']), 
                    triton.cdiv(O, meta['BLOCK_SIZE_O']))
        
        forward_kernel[grid](
            x, weight, output,
            B, I, O,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute backward pass using Triton kernels.

        Args:
            grad_output (torch.Tensor): Gradient tensor of shape (B, O).

        Returns:
            Tuple[torch.Tensor]: Gradients for input (B, I) and weights (I, O).
        """
        x, weight = ctx.saved_tensors
        B, I = x.shape
        O = weight.shape[1]

        # Calculate input gradient
        dx = torch.empty_like(x)
        def grid_dx(meta):
            return (triton.cdiv(B, meta['BLOCK_SIZE_B']),
                    triton.cdiv(I, meta['BLOCK_SIZE_I']))
        
        backward_dx_kernel[grid_dx](
            grad_output, weight, dx,
            B, I, O,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
        )

        # Calculate weight gradient
        dw = torch.empty_like(weight)
        def grid_dw(meta):
            return (triton.cdiv(I, meta['BLOCK_SIZE_I']),
                    triton.cdiv(O, meta['BLOCK_SIZE_O']))
        
        backward_dw_kernel[grid_dw](
            x, grad_output, dw,
            B, I, O,
            x.stride(0), x.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            dw.stride(0), dw.stride(1),
        )

        return dx, dw

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_I': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_I': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_I': 8}, num_warps=4),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def forward_kernel(
    x_ptr, w_ptr, out_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_wi, stride_wo,
    stride_ob, stride_oo,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_I: tl.constexpr,
):
    """
    Triton kernel for forward pass matrix multiplication.

    Args:
        x_ptr (triton.Pointer): Input tensor pointer (B, I).
        w_ptr (triton.Pointer): Weight tensor pointer (I, O).
        out_ptr (triton.Pointer): Output tensor pointer (B, O).
        B, I, O: Problem dimensions.
        Stride parameters for tensor access.
        BLOCK_SIZE_B/I/O: Block sizes for tiling.
        GROUP_SIZE_I: Group size for input dimension blocking.

    Explanation:
        1. Partition problem space into 2D grid over B and O dimensions.
        2. Load blocks of input and weight matrices from global memory.
        3. Perform matrix multiplication with accumulation in FP32.
        4. Store final FP16 result to output tensor.
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, BLOCK_SIZE_I * GROUP_SIZE_I):
        x_offs = i + tl.arange(0, BLOCK_SIZE_I * GROUP_SIZE_I)
        w_offs = i + tl.arange(0, BLOCK_SIZE_I * GROUP_SIZE_I)
        
        x = tl.load(x_ptr + offs_b[:, None] * stride_xb + x_offs[None, :] * stride_xi,
                    mask=(offs_b[:, None] < B) & (x_offs[None, :] < I), other=0.0)
        w = tl.load(w_ptr + w_offs[:, None] * stride_wi + offs_o[None, :] * stride_wo,
                    mask=(w_offs[:, None] < I) & (offs_o[None, :] < O), other=0.0)
        
        x = x.to(tl.float32)
        w = w.to(tl.float32)
        acc += tl.dot(x, w)
    
    acc = acc.to(tl.float16)
    tl.store(out_ptr + offs_b[:, None] * stride_ob + offs_o[None, :] * stride_oo,
             acc, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8}, num_warps=4),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def backward_dx_kernel(
    dout_ptr, w_ptr, dx_ptr,
    B, I, O,
    stride_dob, stride_doo,
    stride_wi, stride_wo,
    stride_dxb, stride_dxi,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    """
    Triton kernel for input gradient calculation.

    Args:
        dout_ptr (triton.Pointer): Output gradient pointer (B, O).
        w_ptr (triton.Pointer): Weight tensor pointer (I, O).
        dx_ptr (triton.Pointer): Input gradient pointer (B, I).
        B, I, O: Problem dimensions.
        Stride parameters for tensor access.
        BLOCK_SIZE_B/I: Block sizes for tiling.
        GROUP_SIZE_B: Group size for batch dimension blocking.

    Explanation:
        1. Partition problem space over B and I dimensions.
        2. Load blocks of output gradients and weight matrix.
        3. Compute gradient through matrix multiplication with weight transpose.
        4. Store computed gradients to input gradient tensor.
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for o in range(0, O, BLOCK_SIZE_B * GROUP_SIZE_B):
        do_offs = o + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
        w_offs = o + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
        
        dout = tl.load(dout_ptr + offs_b[:, None] * stride_dob + do_offs[None, :] * stride_doo,
                       mask=(offs_b[:, None] < B) & (do_offs[None, :] < O), other=0.0)
        w = tl.load(w_ptr + offs_i[:, None] * stride_wi + w_offs[None, :] * stride_wo,
                    mask=(offs_i[:, None] < I) & (w_offs[None, :] < O), other=0.0)
        
        dout = dout.to(tl.float32)
        w = w.to(tl.float32)
        acc += tl.dot(dout, tl.trans(w))
    
    acc = acc.to(tl.float16)
    tl.store(dx_ptr + offs_b[:, None] * stride_dxb + offs_i[None, :] * stride_dxi,
             acc, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_I': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_I': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_I': 8}, num_warps=4),
    ],
    key=['I', 'O', 'B']
)
@triton.jit
def backward_dw_kernel(
    x_ptr, dout_ptr, dw_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_dob, stride_doo,
    stride_dwi, stride_dwo,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_I: tl.constexpr,
):
    """
    Triton kernel for weight gradient calculation.

    Args:
        x_ptr (triton.Pointer): Input tensor pointer (B, I).
        dout_ptr (triton.Pointer): Output gradient pointer (B, O).
        dw_ptr (triton.Pointer): Weight gradient pointer (I, O).
        B, I, O: Problem dimensions.
        Stride parameters for tensor access.
        BLOCK_SIZE_I/O: Block sizes for tiling.
        GROUP_SIZE_I: Group size for input dimension blocking.

    Explanation:
        1. Partition problem space over I and O dimensions.
        2. Load blocks of input and output gradient matrices.
        3. Compute gradient through outer product of input and gradients.
        4. Accumulate and store gradients to weight gradient tensor.
    """
    pid_i = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_O), dtype=tl.float32)
    
    for b in range(0, B, BLOCK_SIZE_I * GROUP_SIZE_I):
        x_offs = b + tl.arange(0, BLOCK_SIZE_I * GROUP_SIZE_I)
        do_offs = b + tl.arange(0, BLOCK_SIZE_I * GROUP_SIZE_I)
        
        x = tl.load(x_ptr + x_offs[:, None] * stride_xb + offs_i[None, :] * stride_xi,
                    mask=(x_offs[:, None] < B) & (offs_i[None, :] < I), other=0.0)
        dout = tl.load(dout_ptr + do_offs[:, None] * stride_dob + offs_o[None, :] * stride_doo,
                       mask=(do_offs[:, None] < B) & (offs_o[None, :] < O), other=0.0)
        
        x = x.to(tl.float32)
        dout = dout.to(tl.float32)
        acc += tl.dot(tl.trans(x), dout)
    
    acc = acc.to(tl.float16)
    tl.store(dw_ptr + offs_i[:, None] * stride_dwi + offs_o[None, :] * stride_dwo,
             acc, mask=(offs_i[:, None] < I) & (offs_o[None, :] < O))

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
