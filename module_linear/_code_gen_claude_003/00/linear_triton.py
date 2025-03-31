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
