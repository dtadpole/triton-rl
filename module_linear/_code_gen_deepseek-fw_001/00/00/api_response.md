Timestamp: 20250406_043953

<think>
Okay, let's look at the error message. The user is getting a ValueError about conflicting meta-parameters: GROUP_SIZE_M. The error message says "Make sure that you don't re-define auto-tuned symbols."

Looking at the code in linear_triton.py, the Triton kernels are using autotune for BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O, etc. But in the kernel calls, like in the forward pass, there's a line that passes GROUP_SIZE_M=8 as an argument.

Wait, the autotune configs in the forward kernel include GROUP_SIZE_M as a parameter. For example, one of the configs is {'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_M': 8}, which is being tuned. But when the kernel is called, the code explicitly sets GROUP_SIZE_M=8. That's a problem because during autotuning, the GROUP_SIZE_M is supposed to be a parameter that's varied, but here it's being fixed in the call, conflicting with the autotuned values.

So the issue is that the kernel is being passed GROUP_SIZE_M as a fixed value in the kernel launch, but it's already part of the autotune configs. Triton's autotuner expects these parameters to be determined by the configs selected during tuning, not overridden when the kernel is called.

The fix would be to remove the GROUP_SIZE_M argument from the kernel calls. That way, the autotuner can manage the GROUP_SIZE_M parameter based on the selected config, and there's no conflict from a fixed value being passed in.

Looking at the code, in the forward function's kernel call:

_forward_kernel[grid](
    input, weight, output,
    B, I, O,
    input.stride(0), input.stride(1),
    weight.stride(0), weight.stride(1),
    output.stride(0), output.stride(1),
    GROUP_SIZE_M=8
)

The last line is passing GROUP_SIZE_M=8. Similarly, in the backward_input_kernel and backward_weight_kernel calls, there's GROUP_SIZE_M=8.

These lines need to be removed because GROUP_SIZE_M is already part of the autotune configurations. By passing it explicitly, it's conflicting with the autotuner's parameters. Removing those arguments should resolve the conflict.

So the fix is to delete the GROUP_SIZE_M=8 parameters from all kernel invocations in the forward and backward methods of the TritonLinearFunction class.

That should allow the autotuner to use the GROUP_SIZE_M values from the configs without being overridden, thus preventing the conflicting meta-parameters error.
</think>

```python
# filename: linear_triton.py

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 256, 'GROUP_SIZE_M': 8}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def _forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_ib, stride_ii,
    stride_wo, stride_wi,
    stride_ob, stride_oo,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Compute forward pass for linear layer using block matrix multiplication.

    Args:
        input_ptr (tl.tensor): Pointer to input tensor of shape (B, I)
        weight_ptr (tl.tensor): Pointer to weight tensor of shape (O, I)
        output_ptr (tl.tensor): Pointer to output tensor of shape (B, O)
        B (int): Batch dimension size
        I (int): Input dimension size
        O (int): Output dimension size
        stride_ib (int): Input batch dimension stride
        stride_ii (int): Input input dimension stride
        stride_wo (int): Weight output dimension stride
        stride_wi (int): Weight input dimension stride
        stride_ob (int): Output batch dimension stride
        stride_oo (int): Output output dimension stride
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_M (tl.constexpr): Number of groups for super-blocking

    Explanation:
        1. Calculate program IDs and block offsets
        2. Load input and weight blocks using matrix pointers
        3. Compute matrix multiplication using dot product
        4. Store results to output tensor
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_n = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_k = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_b = pid_m * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_n * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_ib + offs_i[None, :] * stride_ii
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for k in range(0, num_pid_k):
        mask_input = (offs_b[:, None] < B) & (offs_i[None, :] < I)
        mask_weight = (offs_o[:, None] < O) & (offs_i[None, :] < I)
        
        input_block = tl.load(input_ptrs, mask=mask_input, other=0.0)
        weight_block = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(input_block, weight_block, allow_tf32=True)
        
        input_ptrs += BLOCK_SIZE_I * stride_ii
        weight_ptrs += BLOCK_SIZE_I * stride_wi
    
    mask_output = (offs_b[:, None] < B) & (offs_o[None, :] < O)
    output_ptrs = output_ptr + offs_b[:, None] * stride_ob + offs_o[None, :] * stride_oo
    tl.store(output_ptrs, acc.to(tl.float16), mask=mask_output)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 256, 'GROUP_SIZE_M': 8}, num_warps=8),
    ],
    key=['B', 'O', 'I']
)
@triton.jit
def _backward_input_kernel(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, O, I,
    stride_dob, stride_doo,
    stride_wo, stride_wi,
    stride_dib, stride_dii,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Compute gradient of input for linear layer using matrix multiplication.

    Args:
        d_output_ptr (tl.tensor): Pointer to output gradient tensor of shape (B, O)
        weight_ptr (tl.tensor): Pointer to weight tensor of shape (O, I)
        d_input_ptr (tl.tensor): Pointer to input gradient tensor of shape (B, I)
        B (int): Batch dimension size
        O (int): Output dimension size
        I (int): Input dimension size
        stride_dob (int): Output gradient batch dimension stride
        stride_doo (int): Output gradient output dimension stride
        stride_wo (int): Weight output dimension stride
        stride_wi (int): Weight input dimension stride
        stride_dib (int): Input gradient batch dimension stride
        stride_dii (int): Input gradient input dimension stride
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_M (tl.constexpr): Number of groups for super-blocking

    Explanation:
        1. Calculate program IDs and block offsets
        2. Load output gradient and weight blocks
        3. Compute matrix multiplication with weight transpose
        4. Store results to input gradient tensor
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_n = tl.cdiv(I, BLOCK_SIZE_O)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_b = pid_m * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_n * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    
    d_output_ptrs = d_output_ptr + offs_b[:, None] * stride_dob + offs_o[None, :] * stride_doo
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        mask_dout = (offs_b[:, None] < B) & (offs_o[None, :] < O)
        mask_weight = (offs_o[:, None] < O) & (offs_i[None, :] < I)
        
        d_output_block = tl.load(d_output_ptrs, mask=mask_dout, other=0.0)
        weight_block = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(d_output_block, weight_block, allow_tf32=True)
        
        d_output_ptrs += BLOCK_SIZE_O * stride_doo
        weight_ptrs += BLOCK_SIZE_O * stride_wo
    
    mask_din = (offs_b[:, None] < B) & (offs_i[None, :] < I)
    d_input_ptrs = d_input_ptr + offs_b[:, None] * stride_dib + offs_i[None, :] * stride_dii
    tl.store(d_input_ptrs, acc.to(tl.float16), mask=mask_din)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 256, 'GROUP_SIZE_M': 8}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def _backward_weight_kernel(
    input_ptr, d_output_ptr, d_weight_ptr,
    B, I, O,
    stride_ib, stride_ii,
    stride_dob, stride_doo,
    stride_dwo, stride_dwi,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Compute gradient of weight for linear layer using outer product.

    Args:
        input_ptr (tl.tensor): Pointer to input tensor of shape (B, I)
        d_output_ptr (tl.tensor): Pointer to output gradient tensor of shape (B, O)
        d_weight_ptr (tl.tensor): Pointer to weight gradient tensor of shape (O, I)
        B (int): Batch dimension size
        I (int): Input dimension size
        O (int): Output dimension size
        stride_ib (int): Input batch dimension stride
        stride_ii (int): Input input dimension stride
        stride_dob (int): Output gradient batch dimension stride
        stride_doo (int): Output gradient output dimension stride
        stride_dwo (int): Weight gradient output dimension stride
        stride_dwi (int): Weight gradient input dimension stride
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        GROUP_SIZE_M (tl.constexpr): Number of groups for super-blocking

    Explanation:
        1. Calculate program IDs and block offsets
        2. Load input and output gradient blocks
        3. Compute outer product accumulation
        4. Atomically store results to weight gradient tensor
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(O, BLOCK_SIZE_B)
    num_pid_n = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_o = pid_m * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_n * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = tl.arange(0, BLOCK_SIZE_B)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_ib + offs_i[None, :] * stride_ii
    d_output_ptrs = d_output_ptr + offs_b[:, None] * stride_dob + offs_o[None, :] * stride_doo
    
    acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_B), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        mask_input = (offs_b[:, None] < B) & (offs_i[None, :] < I)
        mask_dout = (offs_b[:, None] < B) & (offs_o[None, :] < O)
        
        input_block = tl.load(input_ptrs, mask=mask_input, other=0.0)
        d_output_block = tl.load(d_output_ptrs, mask=mask_dout, other=0.0)
        
        acc += tl.dot(input_block.T, d_output_block, allow_tf32=True)
        
        input_ptrs += BLOCK_SIZE_B * stride_ib
        d_output_ptrs += BLOCK_SIZE_B * stride_dob
    
    mask_dw = (offs_o[:, None] < O) & (offs_i[None, :] < I)
    d_weight_ptrs = d_weight_ptr + offs_o[:, None] * stride_dwo + offs_i[None, :] * stride_dwi
    tl.atomic_add(d_weight_ptrs, acc.T.to(tl.float16), mask=mask_dw)


class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O, _ = weight.shape
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(O, meta['BLOCK_SIZE_O']),)
        _forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1)
        )
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O, _ = weight.shape
        
        d_input = torch.empty_like(input)
        d_weight = torch.zeros_like(weight)
        
        # Compute input gradient
        grid_input = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_O']),)
        _backward_input_kernel[grid_input](
            d_output, weight, d_input,
            B, O, I,
            d_output.stride(0), d_output.stride(1),
            weight.stride(0), weight.stride(1),
            d_input.stride(0), d_input.stride(1)
        )
        
        # Compute weight gradient
        grid_weight = lambda meta: (triton.cdiv(O, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_I']),)
        _backward_weight_kernel[grid_weight](
            input, d_output, d_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            d_output.stride(0), d_output.stride(1),
            d_weight.stride(0), d_weight.stride(1)
        )
        
        return d_input, d_weight


class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
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
```