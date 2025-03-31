Timestamp: 20250331_040805

```python
# filename: linear_triton.py
import torch
import triton
import triton.language as tl
import math

# Helper function
def ceil_div(a, b):
    return (a + b - 1) // b

# -------------------\
# Triton Kernels
# -------------------\

@triton.autotune(
    configs=[
        # Basic configs
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1, 'num_stages': 1, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1, 'num_stages': 4, 'num_warps': 8}),
        # Configs with Group Sizes (Super Blocks)
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_O': 1, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 4, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_O': 2, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_O': 2, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_forward_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, output_ptr,
    # Matrix dimensions
    B, I, O,
    # Matrix strides
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_O: tl.constexpr
):
    """
    [Computes the forward pass of a linear layer: output = input @ weight.T]

    Args:
        input_ptr (tl.pointer_type): Pointer to the input tensor. [B, I]
        weight_ptr (tl.pointer_type): Pointer to the weight tensor. [O, I]
        output_ptr (tl.pointer_type): Pointer to the output tensor. [B, O]
        B (int): Batch size.
        I (int): Input features dimension.
        O (int): Output features dimension.
        stride_input_b (int): Stride of the input tensor along the batch dimension.
        stride_input_i (int): Stride of the input tensor along the input feature dimension.
        stride_weight_o (int): Stride of the weight tensor along the output feature dimension.
        stride_weight_i (int): Stride of the weight tensor along the input feature dimension.
        stride_output_b (int): Stride of the output tensor along the batch dimension.
        stride_output_o (int): Stride of the output tensor along the output feature dimension.
        BLOCK_SIZE_B (tl.constexpr): Tile size for the batch dimension.
        BLOCK_SIZE_I (tl.constexpr): Tile size for the input feature dimension (reduction dimension K).
        BLOCK_SIZE_O (tl.constexpr): Tile size for the output feature dimension.
        GROUP_SIZE_B (tl.constexpr): Grouping factor for the batch dimension (super blocks).
        GROUP_SIZE_O (tl.constexpr): Grouping factor for the output dimension (super blocks).

    Returns:
        None: The result is written directly to output_ptr.

    Explanation:
        1. Calculate the program IDs (`pid_b`, `pid_o`) for the current instance along the B and O dimensions based on the grid configuration. The grid is calculated outside the kernel.
        2. Calculate the starting indices (`start_b`, `start_o`) for the group block this program is responsible for.
        3. Iterate through the group tiles assigned to this instance (`gb`, `go`).
        4. Calculate the base offsets (`offset_b`, `offset_o`) for the current sub-tile within the group.
        5. Initialize an accumulator `acc` matrix of size `(BLOCK_SIZE_B, BLOCK_SIZE_O)` with zeros (using float32 for precision).
        6. Iterate over the input feature dimension `I` in blocks of `BLOCK_SIZE_I`.
        7. Calculate pointers (`a_ptrs`, `b_ptrs`) to the current blocks of the input (`a`) and weight (`b`) tensors.
        8. Define boundary masks (`a_mask`, `b_mask`) for the `I` dimension to handle cases where `I` is not divisible by `BLOCK_SIZE_I`.
        9. Load the input block `a` (`BLOCK_SIZE_B, BLOCK_SIZE_I`) and weight block `b` (`BLOCK_SIZE_O, BLOCK_SIZE_I`) using masks.
        10. Compute the matrix multiplication `a @ b.T` for the current blocks and add the result to the accumulator `acc`. Use `tl.dot` with `allow_tf32=True`. Note `b` needs transposing.
        11. After iterating through `I`, calculate the pointers (`c_ptrs`) to the corresponding block in the output tensor.
        12. Define boundary masks (`c_mask`) for the `B` and `O` dimensions.
        13. Store the accumulated result `acc` (cast to the output tensor's dtype) into the output tensor using the mask.
    """
    # 1. Program IDs
    pid = tl.program_id(axis=0) # We use a 1D launch grid. Program ID identifies the B-O block.
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B * GROUP_SIZE_B)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O * GROUP_SIZE_O)
    pid_b = pid // num_pid_o
    pid_o = pid % num_pid_o

    # 2. Iterate through group tiles (handled by GROUP_SIZE_* within the loop)
    for gb in range(GROUP_SIZE_B):
        actual_pid_b = pid_b * GROUP_SIZE_B + gb
        for go in range(GROUP_SIZE_O):
            actual_pid_o = pid_o * GROUP_SIZE_O + go

            # 3. Base offsets for the current sub-tile
            offset_b = actual_pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
            offset_o = actual_pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

            # 4. Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)

            # 5. Iterate over K (I dimension)
            for k in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
                offset_i = k * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

                # 6. Calculate pointers for current blocks
                a_ptrs = input_ptr + offset_b[:, None] * stride_input_b + offset_i[None, :] * stride_input_i
                # Weight is (O, I), need (BLOCK_O, BLOCK_I) block -> transpose for dot
                b_ptrs = weight_ptr + offset_o[:, None] * stride_weight_o + offset_i[None, :] * stride_weight_i

                # 7. Define boundary masks for K dimension
                a_mask = (offset_b[:, None] < B) & (offset_i[None, :] < I)
                b_mask = (offset_o[:, None] < O) & (offset_i[None, :] < I)

                # 8. Load blocks with masking
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0) # Shape (BLOCK_SIZE_O, BLOCK_SIZE_I)

                # 9. Compute matrix multiplication and accumulate
                # a: (BLOCK_B, BLOCK_I), b: (BLOCK_O, BLOCK_I) -> need b.T
                # Triton dot(A(M,K), B(N,K).T) computes A @ B.T
                acc += tl.dot(a, tl.trans(b), allow_tf32=True)

            # 10. Calculate output pointers
            c_ptrs = output_ptr + offset_b[:, None] * stride_output_b + offset_o[None, :] * stride_output_o

            # 11. Define boundary masks for M (B) and N (O) dimensions
            c_mask = (offset_b[:, None] < B) & (offset_o[None, :] < O)

            # 12. Store the result
            tl.store(c_ptrs, acc.to(output_ptr.dtype.element_ty), mask=c_mask)


@triton.autotune(
    configs=[
        # Basic configs
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'num_stages': 1, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1, 'num_stages': 4, 'num_warps': 8}),
         # Configs with Group Sizes (Super Blocks)
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_I': 1, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 4, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_I': 2, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_I': 2, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['B', 'O', 'I'],
)
@triton.jit
def linear_backward_dx_kernel(
    # Pointers to matrices
    d_output_ptr, weight_ptr, dx_ptr,
    # Matrix dimensions
    B, O, I,
    # Matrix strides
    stride_d_output_b, stride_d_output_o,
    stride_weight_o, stride_weight_i,
    stride_dx_b, stride_dx_i,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_I: tl.constexpr
):
    """
    [Computes the backward pass for the input gradient (dx) of a linear layer: dx = d_output @ weight]

    Args:
        d_output_ptr (tl.pointer_type): Pointer to the gradient of the output tensor. [B, O]
        weight_ptr (tl.pointer_type): Pointer to the weight tensor. [O, I]
        dx_ptr (tl.pointer_type): Pointer to the input gradient tensor (output). [B, I]
        B (int): Batch size.
        O (int): Output features dimension (reduction dimension K).
        I (int): Input features dimension.
        stride_d_output_b (int): Stride of d_output along the batch dimension.
        stride_d_output_o (int): Stride of d_output along the output feature dimension.
        stride_weight_o (int): Stride of weight along the output feature dimension.
        stride_weight_i (int): Stride of weight along the input feature dimension.
        stride_dx_b (int): Stride of dx along the batch dimension.
        stride_dx_i (int): Stride of dx along the input feature dimension.
        BLOCK_SIZE_B (tl.constexpr): Tile size for the batch dimension.
        BLOCK_SIZE_O (tl.constexpr): Tile size for the output feature dimension (reduction dimension K).
        BLOCK_SIZE_I (tl.constexpr): Tile size for the input feature dimension.
        GROUP_SIZE_B (tl.constexpr): Grouping factor for the batch dimension (super blocks).
        GROUP_SIZE_I (tl.constexpr): Grouping factor for the input dimension (super blocks).

    Returns:
        None: The result is written directly to dx_ptr.

    Explanation:
        1. Calculate the program IDs (`pid_b`, `pid_i`) for the current instance along the B and I dimensions based on the grid configuration.
        2. Iterate through the group tiles assigned to this instance (`gb`, `gi`).
        3. Calculate the base offsets (`offset_b`, `offset_i`) for the current sub-tile within the group.
        4. Initialize an accumulator `acc` matrix of size `(BLOCK_SIZE_B, BLOCK_SIZE_I)` with zeros (using float32 for precision).
        5. Iterate over the output feature dimension `O` (reduction dim K) in blocks of `BLOCK_SIZE_O`.
        6. Calculate pointers (`a_ptrs`, `b_ptrs`) to the current blocks of d_output (`a`) and weight (`b`).
        7. Define boundary masks (`a_mask`, `b_mask`) for the `O` dimension.
        8. Load the d_output block `a` (`BLOCK_SIZE_B, BLOCK_SIZE_O`) and weight block `b` (`BLOCK_SIZE_O, BLOCK_SIZE_I`) using masks.
        9. Compute the matrix multiplication `a @ b` and add the result to the accumulator `acc`. Use `tl.dot` with `allow_tf32=True`.
        10. After iterating through `O`, calculate the pointers (`c_ptrs`) to the corresponding block in the `dx` tensor.
        11. Define boundary masks (`c_mask`) for the `B` and `I` dimensions.
        12. Store the accumulated result `acc` (cast to the dx tensor's dtype) into the `dx` tensor using the mask.
    """
    # 1. Program IDs
    pid = tl.program_id(axis=0) # 1D launch grid
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B * GROUP_SIZE_B)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I * GROUP_SIZE_I)
    pid_b = pid // num_pid_i
    pid_i = pid % num_pid_i

    # 2. Iterate through group tiles
    for gb in range(GROUP_SIZE_B):
        actual_pid_b = pid_b * GROUP_SIZE_B + gb
        for gi in range(GROUP_SIZE_I):
            actual_pid_i = pid_i * GROUP_SIZE_I + gi

            # 3. Base offsets for the current sub-tile
            offset_b = actual_pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
            offset_i = actual_pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

            # 4. Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)

            # 5. Iterate over K (O dimension)
            for k in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
                offset_o = k * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

                # 6. Calculate pointers for current blocks
                a_ptrs = d_output_ptr + offset_b[:, None] * stride_d_output_b + offset_o[None, :] * stride_d_output_o
                b_ptrs = weight_ptr + offset_o[:, None] * stride_weight_o + offset_i[None, :] * stride_weight_i

                # 7. Define boundary masks for K dimension
                a_mask = (offset_b[:, None] < B) & (offset_o[None, :] < O)
                b_mask = (offset_o[:, None] < O) & (offset_i[None, :] < I)

                # 8. Load blocks with masking
                a = tl.load(a_ptrs, mask=a_mask, other=0.0) # Shape (BLOCK_B, BLOCK_O)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0) # Shape (BLOCK_O, BLOCK_I)

                # 9. Compute matrix multiplication and accumulate
                acc += tl.dot(a, b, allow_tf32=True)

            # 10. Calculate output pointers
            c_ptrs = dx_ptr + offset_b[:, None] * stride_dx_b + offset_i[None, :] * stride_dx_i

            # 11. Define boundary masks for M (B) and N (I) dimensions
            c_mask = (offset_b[:, None] < B) & (offset_i[None, :] < I)

            # 12. Store the result
            tl.store(c_ptrs, acc.to(dx_ptr.dtype.element_ty), mask=c_mask)


@triton.autotune(
    configs=[
        # Basic configs
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'num_stages': 1, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1, 'num_stages': 4, 'num_warps': 8}),
        # Configs with Group Sizes (Super Blocks)
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 4, 'GROUP_SIZE_I': 1, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 4, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_O': 2, 'GROUP_SIZE_I': 2, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_O': 2, 'GROUP_SIZE_I': 2, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['O', 'B', 'I'], # Note: Key order matters for autotuning lookup
)
@triton.jit
def linear_backward_dw_kernel(
    # Pointers to matrices
    d_output_ptr, input_ptr, dw_ptr,
    # Matrix dimensions
    B, O, I,
    # Matrix strides
    stride_d_output_b, stride_d_output_o,
    stride_input_b, stride_input_i,
    stride_dw_o, stride_dw_i,
    # Meta-parameters
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr
):
    """
    [Computes the backward pass for the weight gradient (dw) of a linear layer: dw = d_output.T @ input]

    Args:
        d_output_ptr (tl.pointer_type): Pointer to the gradient of the output tensor. [B, O]
        input_ptr (tl.pointer_type): Pointer to the input tensor. [B, I]
        dw_ptr (tl.pointer_type): Pointer to the weight gradient tensor (output). [O, I]
        B (int): Batch size (reduction dimension K).
        O (int): Output features dimension.
        I (int): Input features dimension.
        stride_d_output_b (int): Stride of d_output along the batch dimension.
        stride_d_output_o (int): Stride of d_output along the output feature dimension.
        stride_input_b (int): Stride of input along the batch dimension.
        stride_input_i (int): Stride of input along the input feature dimension.
        stride_dw_o (int): Stride of dw along the output feature dimension.
        stride_dw_i (int): Stride of dw along the input feature dimension.
        BLOCK_SIZE_O (tl.constexpr): Tile size for the output feature dimension.
        BLOCK_SIZE_B (tl.constexpr): Tile size for the batch dimension (reduction dimension K).
        BLOCK_SIZE_I (tl.constexpr): Tile size for the input feature dimension.
        GROUP_SIZE_O (tl.constexpr): Grouping factor for the output dimension (super blocks).
        GROUP_SIZE_I (tl.constexpr): Grouping factor for the input dimension (super blocks).

    Returns:
        None: The result is written directly to dw_ptr.

    Explanation:
        1. Calculate the program IDs (`pid_o`, `pid_i`) for the current instance along the O and I dimensions based on the grid configuration.
        2. Iterate through the group tiles assigned to this instance (`go`, `gi`).
        3. Calculate the base offsets (`offset_o`, `offset_i`) for the current sub-tile within the group.
        4. Initialize an accumulator `acc` matrix of size `(BLOCK_SIZE_O, BLOCK_SIZE_I)` with zeros (using float32 for precision).
        5. Iterate over the batch dimension `B` (reduction dim K) in blocks of `BLOCK_SIZE_B`.
        6. Calculate pointers (`a_ptrs`, `b_ptrs`) to the current blocks of d_output (`a`) and input (`b`).
        7. Define boundary masks (`a_mask`, `b_mask`) for the `B` dimension.
        8. Load the d_output block `a` (`BLOCK_SIZE_B, BLOCK_SIZE_O`) and input block `b` (`BLOCK_SIZE_B, BLOCK_SIZE_I`) using masks.
        9. Compute the matrix multiplication `a.T @ b` and add the result to the accumulator `acc`. Use `tl.dot` with `allow_tf32=True`. Note `a` needs transposing.
        10. After iterating through `B`, calculate the pointers (`c_ptrs`) to the corresponding block in the `dw` tensor.
        11. Define boundary masks (`c_mask`) for the `O` and `I` dimensions.
        12. Store the accumulated result `acc` (cast to the dw tensor's dtype) into the `dw` tensor using the mask. Atomics are not needed here as accumulation happens locally within the kernel instance before the final store.
    """
    # 1. Program IDs
    pid = tl.program_id(axis=0) # 1D launch grid
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O * GROUP_SIZE_O)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I * GROUP_SIZE_I)
    pid_o = pid // num_pid_i
    pid_i = pid % num_pid_i

    # 2. Iterate through group tiles
    for go in range(GROUP_SIZE_O):
        actual_pid_o = pid_o * GROUP_SIZE_O + go
        for gi in range(GROUP_SIZE_I):
            actual_pid_i = pid_i * GROUP_SIZE_I + gi

            # 3. Base offsets for the current sub-tile
            offset_o = actual_pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
            offset_i = actual_pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

            # 4. Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)

            # 5. Iterate over K (B dimension)
            for k in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
                offset_b = k * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)

                # 6. Calculate pointers for current blocks
                # d_output (B, O) -> load block (BLOCK_B, BLOCK_O)
                a_ptrs = d_output_ptr + offset_b[:, None] * stride_d_output_b + offset_o[None, :] * stride_d_output_o
                # input (B, I) -> load block (BLOCK_B, BLOCK_I)
                b_ptrs = input_ptr + offset_b[:, None] * stride_input_b + offset_i[None, :] * stride_input_i

                # 7. Define boundary masks for K dimension
                a_mask = (offset_b[:, None] < B) & (offset_o[None, :] < O)
                b_mask = (offset_b[:, None] < B) & (offset_i[None, :] < I)

                # 8. Load blocks with masking
                a = tl.load(a_ptrs, mask=a_mask, other=0.0) # Shape (BLOCK_B, BLOCK_O)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0) # Shape (BLOCK_B, BLOCK_I)

                # 9. Compute matrix multiplication and accumulate
                # Need a.T @ b = (BLOCK_O, BLOCK_B) @ (BLOCK_B, BLOCK_I) -> (BLOCK_O, BLOCK_I)
                acc += tl.dot(tl.trans(a), b, allow_tf32=True)

            # 10. Calculate output pointers
            c_ptrs = dw_ptr + offset_o[:, None] * stride_dw_o + offset_i[None, :] * stride_dw_i

            # 11. Define boundary masks for M (O) and N (I) dimensions
            c_mask = (offset_o[:, None] < O) & (offset_i[None, :] < I)

            # 12. Store the result
            tl.store(c_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=c_mask)


# -------------------\
# Autograd Function
# -------------------\

class _Linear(torch.autograd.Function):
    """
    [Custom autograd function for Triton-based Linear layer.]

    Explanation:
        1. forward: Computes the forward pass using the `linear_forward_kernel`.
        2. backward: Computes the backward pass using `linear_backward_dx_kernel` and `linear_backward_dw_kernel`.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        [Computes the forward pass of the linear layer (output = input @ weight.T).]

        Args:
            ctx (torch.autograd.function.Context): Context object to save tensors for backward.
            input (torch.Tensor): Input tensor. [B, I]
            weight (torch.Tensor): Weight tensor. [O, I]

        Returns:
            torch.Tensor: Output tensor. [B, O]

        Explanation:
            1. Check contiguity of input and weight tensors, make contiguous if necessary.
            2. Get dimensions B, I from input and O from weight.
            3. Create an empty output tensor of shape [B, O] with the correct dtype and device.
            4. Define the grid size calculation function for the Triton kernel. It uses a 1D grid where each program handles a block group.
            5. Launch the `linear_forward_kernel` with the necessary pointers, dimensions, strides, and let the autotuner handle block/group sizes.
            6. Save the input and weight tensors for the backward pass using `ctx.save_for_backward`.
            7. Return the output tensor.
        """
        # 1. Ensure contiguity
        input_c = input.contiguous()
        weight_c = weight.contiguous()

        # 2. Get dimensions
        B, I = input_c.shape
        O, _ = weight_c.shape
        assert _ == I, f"Weight shape[1] ({_}) must match input shape[1] ({I})"

        # 3. Create output tensor
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)

        # 4. Define grid size function
        def grid(meta):
            # Calculates grid size based on block/group sizes from autotuner meta
            # We use a 1D grid, where each program handles a group of blocks.
            return (ceil_div(B, meta['BLOCK_SIZE_B'] * meta['GROUP_SIZE_B']) *
                    ceil_div(O, meta['BLOCK_SIZE_O'] * meta['GROUP_SIZE_O']), )

        # 5. Launch kernel
        linear_forward_kernel[grid](
            input_c, weight_c, output,
            B, I, O,
            input_c.stride(0), input_c.stride(1),
            weight_c.stride(0), weight_c.stride(1),
            output.stride(0), output.stride(1),
            # Autotuner provides BLOCK_SIZE_* and GROUP_SIZE_* based on keys B, I, O
        )

        # 6. Save for backward
        ctx.save_for_backward(input_c, weight_c)

        # 7. Return output
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        [Computes the backward pass for input and weight gradients.]

        Args:
            ctx (torch.autograd.function.Context): Context object with saved tensors.
            grad_output (torch.Tensor): Gradient of the loss w.r.t. the output tensor. [B, O]

        Returns:
            tuple[torch.Tensor | None, torch.Tensor | None]: Gradients w.r.t. input and weight. (dx, dw)

        Explanation:
            1. Retrieve saved input and weight tensors.
            2. Check contiguity of grad_output, make contiguous if necessary.
            3. Get dimensions B, I, O.
            4. --- Calculate dx (input gradient: dx = grad_output @ weight) ---
            5. Create an empty dx tensor of shape [B, I].
            6. Define the grid size calculation function for the `linear_backward_dx_kernel`.
            7. Launch the `linear_backward_dx_kernel`.
            8. --- Calculate dw (weight gradient: dw = grad_output.T @ input) ---
            9. Create an empty dw tensor of shape [O, I].
            10. Define the grid size calculation function for the `linear_backward_dw_kernel`.
            11. Launch the `linear_backward_dw_kernel`.
            12. Return the computed gradients (dx, dw). Return None for bias gradient as it's not used.
        """
        # 1. Retrieve saved tensors
        input, weight = ctx.saved_tensors

        # 2. Ensure contiguity for grad_output
        grad_output_c = grad_output.contiguous()

        # 3. Get dimensions
        B, I = input.shape
        O, _ = weight.shape
        assert grad_output_c.shape == (B, O)

        # 4. --- Calculate dx ---
        # 5. Create dx tensor
        dx = torch.empty_like(input)

        # 6. Define grid for dx kernel
        def grid_dx(meta):
            # 1D grid: Each program handles a group of blocks for (B, I) output.
            return (ceil_div(B, meta['BLOCK_SIZE_B'] * meta['GROUP_SIZE_B']) *
                    ceil_div(I, meta['BLOCK_SIZE_I'] * meta['GROUP_SIZE_I']), )

        # 7. Launch dx kernel
        linear_backward_dx_kernel[grid_dx](
            grad_output_c, weight, dx,
            B, O, I,
            grad_output_c.stride(0), grad_output_c.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
            # Autotuner provides BLOCK_SIZE_* and GROUP_SIZE_* based on keys B, O, I
        )

        # 8. --- Calculate dw ---
        # 9. Create dw tensor
        dw = torch.empty_like(weight)

        # 10. Define grid for dw kernel
        def grid_dw(meta):
            # 1D grid: Each program handles a group of blocks for (O, I) output.
            return (ceil_div(O, meta['BLOCK_SIZE_O'] * meta['GROUP_SIZE_O']) *
                    ceil_div(I, meta['BLOCK_SIZE_I'] * meta['GROUP_SIZE_I']), )

        # 11. Launch dw kernel
        linear_backward_dw_kernel[grid_dw](
            grad_output_c, input, dw,
            B, O, I,
            grad_output_c.stride(0), grad_output_c.stride(1),
            input.stride(0), input.stride(1),
            dw.stride(0), dw.stride(1),
            # Autotuner provides BLOCK_SIZE_* and GROUP_SIZE_* based on keys O, B, I
        )

        # 12. Return gradients
        return dx, dw


# -------------------\
# PyTorch Module
# -------------------\

class TritonLinear(torch.nn.Module):
    """
    [Triton implementation of a standard linear layer (torch.nn.Linear without bias).]

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        device (str or torch.device): The desired device of the parameters.
        dtype (torch.dtype): The desired floating point type of the parameters.

    Attributes:
        weight (torch.nn.Parameter): The learnable weights of the module of shape (out_features, in_features).

    Explanation:
        1. Initialize the nn.Module superclass.
        2. Store in_features and out_features.
        3. Create the weight parameter tensor with shape (out_features, in_features) on the specified device and dtype.
        4. Initialize the weight parameter using Kaiming uniform initialization.
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        [Initializes the TritonLinear module.]

        Args:
            in_features (int): Size of each input sample (I).
            out_features (int): Size of each output sample (O).
            device (str or torch.device, optional): The target device for the weight tensor. Defaults to None (CPU).
            dtype (torch.dtype, optional): The target data type for the weight tensor. Defaults to None (torch default).

        Returns:
            None

        Explanation:
            1. Call the superclass constructor.
            2. Store input and output feature sizes.
            3. Create the weight parameter `self.weight` as a `torch.nn.Parameter` of shape `(out_features, in_features)`.
            4. Initialize weights using `torch.nn.init.kaiming_uniform_` with `a=math.sqrt(5)`.\n        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # Initialize weights similar to torch.nn.Linear
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        [Performs the forward pass using the custom Triton autograd function.]

        Args:
            input (torch.Tensor): The input tensor. [B, in_features]

        Returns:
            torch.Tensor: The output tensor. [B, out_features]

        Explanation:
            1. Call the `_Linear.apply` method, passing the input tensor and the module's weight parameter.\n        """
        return _Linear.apply(input, self.weight)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=False'

# -------------------\
# Benchmark Code
# -------------------\

# The following code is provided by the user for testing correctness and performance.
if __name__ == "__main__":
    # Ensure benchmark.py and _benchmark_util.py are in the same directory or PYTHONPATH
    try:
        from benchmark import (
            verify_correctness_forward, benchmark_performance_forward,
            verify_correctness_backward, benchmark_performance_backward
        )
    except ImportError:
        print("Error: benchmark.py not found. Please ensure benchmark.py and _benchmark_util.py are accessible.")
        print("You might need to copy them into the current directory.")
        exit(1)

    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    # Use float16 for benchmarking as requested
    dtype = torch.float16
    device = 'cuda'

    print(f"Running benchmarks with dtype={dtype} on device={device}")
    print(f"B sizes: {B_sizes}")
    print(f"I sizes: {I_sizes}")
    print(f"O sizes: {O_sizes}")
    print("-" * 30)

    def module_generator(B, I, O, device=device, dtype=dtype):
        # Note: Using the class defined in *this* file
        triton_module = TritonLinear(I, O, device=device, dtype=dtype)
        torch_module = torch.nn.Linear(I, O, bias=False, device=device, dtype=dtype)
        # Ensure weights are identical for fair comparison
        with torch.no_grad():
            triton_module.weight.copy_(torch_module.weight)
        return triton_module, torch_module

    def forward_generator(B, I, O, device=device, dtype=dtype):
        # Using normalized inputs as in the request
        input_tensor = torch.randn(B, I, device=device, dtype=dtype, requires_grad=True) # Ensure requires_grad=True for backward pass later
        input_tensor = torch.nn.functional.normalize(input_tensor, dim=-1)
        # Ensure inputs is a tuple as expected by benchmark functions
        return (input_tensor,) # Return as tuple

    print("Verifying Forward Pass Correctness...")
    # Relaxed tolerance for FP16, especially with TF32 potentially enabled in matmuls
    verify_correctness_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes], dtype=dtype, atol=5e-2, rtol=5e-2)
    print("\nBenchmarking Forward Pass Performance...")
    benchmark_performance_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes], dtype=dtype)

    def backward_generator(B, I, O, device=device, dtype=dtype):
        # Need requires_grad=True on input for backward pass
        input_tensor = torch.randn(B, I, device=device, dtype=dtype, requires_grad=True)
        input_tensor = torch.nn.functional.normalize(input_tensor, dim=-1)
        # Using normalized d_output as in the request
        d_output = torch.randn(B, O, device=device, dtype=dtype)
        d_output = torch.nn.functional.normalize(d_output, dim=-1)
        # Ensure inputs is a tuple as expected by benchmark functions
        return (input_tensor,), d_output # Return input as tuple

    print("\nVerifying Backward Pass Correctness...")
    # Relaxed tolerance for FP16, especially with TF32 potentially enabled in matmuls
    verify_correctness_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes], dtype=dtype, atol=5e-2, rtol=5e-2)
    print("\nBenchmarking Backward Pass Performance...")
    benchmark_performance_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes], dtype=dtype)

    print("\nBenchmark finished.")

```