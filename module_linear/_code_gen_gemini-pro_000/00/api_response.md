Timestamp: 20250331_040632

```think
## Problem Decomposition

1.  **Goal:** Implement a Triton equivalent of `torch.nn.Linear` (without bias), including forward and backward passes.
2.  **Components:**
    *   A `torch.nn.Module` (`TritonLinear`) to hold the weight parameter and define the forward pass structure.
    *   A `torch.autograd.Function` (`_Linear`) to encapsulate the Triton kernel calls for forward and backward, enabling gradient calculation.
    *   A Triton kernel for the forward pass (`linear_forward_kernel`).
    *   Two Triton kernels for the backward pass:
        *   One for the input gradient (`dx`) (`linear_backward_dx_kernel`).
        *   One for the weight gradient (`dw`) (`linear_backward_dw_kernel`).
3.  **Requirements:**
    *   Use Triton for kernel implementations.
    *   Implement both forward (`output = input @ weight.T`) and backward (`dx = d_output @ weight`, `dw = d_output.T @ input`).
    *   Use `@triton.autotune` for block sizes (`BLOCK_SIZE_B`, `BLOCK_SIZE_I`, `BLOCK_SIZE_O` - adjusting names based on kernel dimensions) in all kernels.
    *   Implement super blocks using `GROUP_SIZE_*` parameters and include them in the autotuning.
    *   Provide detailed docstrings in the specified format for all Python functions and Triton kernels.
    *   Use the provided `benchmark.py` functions and the exact `if __name__ == "__main__":` block for testing correctness and performance.
    *   Assume `benchmark.py` exists.
    *   Target `torch.float16` dtype.
    *   Handle potential non-divisibility of dimensions by block sizes using masking.
    *   Use `tl.float32` for accumulation in GEMMs for precision.

## Implementation Details

### `TritonLinear` Module

*   Inherits `torch.nn.Module`.
*   `__init__`: Takes `in_features`, `out_features`, `device`, `dtype`. Initializes `self.weight` as a `torch.nn.Parameter` with shape `(out_features, in_features)`. Initializes weights using Kaiming uniform, similar to `torch.nn.Linear`.
*   `forward`: Calls `_Linear.apply(input, self.weight)`.

### `_Linear` Autograd Function

*   `forward(ctx, input, weight)`:
    1.  Validate input shapes.
    2.  Prepare output tensor `output` with shape `(B, O)`.
    3.  Define grid calculation based on `B`, `O`, `BLOCK_SIZE_B`, `BLOCK_SIZE_O`, `GROUP_SIZE_B`, `GROUP_SIZE_O`.
    4.  Call `linear_forward_kernel[grid](...)` passing input, weight, output, dimensions, and strides.
    5.  `ctx.save_for_backward(input, weight)`
    6.  Return `output`.
*   `backward(ctx, grad_output)`:
    1.  Retrieve `input`, `weight` using `ctx.saved_tensors`.
    2.  **Calculate `dx` (gradient w.r.t. input):**
        *   Prepare `dx` tensor with shape `(B, I)`.
        *   Define grid calculation for `dx` based on `B`, `I`, `BLOCK_SIZE_B`, `BLOCK_SIZE_I`, `GROUP_SIZE_B`, `GROUP_SIZE_I`.
        *   Call `linear_backward_dx_kernel[grid](...)` passing `grad_output`, `weight`, `dx`, dimensions, and strides.
    3.  **Calculate `dw` (gradient w.r.t. weight):**
        *   Prepare `dw` tensor with shape `(O, I)`.
        *   Define grid calculation for `dw` based on `O`, `I`, `BLOCK_SIZE_O`, `BLOCK_SIZE_I`, `GROUP_SIZE_O`, `GROUP_SIZE_I`.
        *   Call `linear_backward_dw_kernel[grid](...)` passing `grad_output`, `input`, `dw`, dimensions, and strides.
    4.  Return `dx`, `dw`.

### Triton Kernels

**General Kernel Structure:**

*   Use `@triton.jit`.
*   Define autotune config with relevant `BLOCK_SIZE_*` and `GROUP_SIZE_*`. Use powers of 2 for block sizes (e.g., 16, 32, 64, 128) and small integers for group sizes (e.g., 1, 2, 4). Include `num_warps` and `num_stages`.
*   Get dimensions (`B`, `I`, `O`) and strides from arguments.
*   Calculate program IDs (`pid_b`, `pid_i`, `pid_o` based on kernel).
*   Implement outer loops for `GROUP_SIZE_*`.
*   Inside group loops, calculate base offsets for the current sub-block using `pid_*`, `group_idx_*`, and `BLOCK_SIZE_*`.
*   Implement inner loop for the reduction dimension (`K`).
*   Inside the inner loop:
    *   Calculate pointers (`*_ptrs`) for the current blocks of operands A and B using offsets and `tl.arange`.
    *   Create masks for boundary handling.
    *   Load data using `tl.load` with masks and `other=0.0`.
    *   Perform `tl.dot` operation, potentially with `tl.trans`, accumulating in `tl.float32`. Set `allow_tf32=True`.
*   After the inner loop, calculate output pointers (`c_ptrs`) using offsets.
*   Create output masks.
*   Store the final accumulated result (`acc`) cast back to the output dtype using `tl.store` with masks. (No atomics needed if each program instance computes disjoint output blocks).

**`linear_forward_kernel` (Output = Input @ Weight.T)**

*   Inputs: `input_ptr`, `weight_ptr`, `output_ptr`, dimensions (`B`, `I`, `O`), strides.
*   Autotune Keys: `BLOCK_SIZE_B`, `BLOCK_SIZE_I`, `BLOCK_SIZE_O`, `GROUP_SIZE_B`, `GROUP_SIZE_O`.
*   Grid dims: `(ceil_div(B, BLOCK_SIZE_B * GROUP_SIZE_B), ceil_div(O, BLOCK_SIZE_O * GROUP_SIZE_O))`.
*   `pid_b = tl.program_id(0)`, `pid_o = tl.program_id(1)`.
*   Outer loops: `gb` over `GROUP_SIZE_B`, `go` over `GROUP_SIZE_O`.
*   Offsets: `offset_b`, `offset_o`.
*   Inner loop: `k` over `I` in steps of `BLOCK_SIZE_I`.
*   Load `a` (input): `(BLOCK_SIZE_B, BLOCK_SIZE_I)`.
*   Load `b` (weight): `(BLOCK_SIZE_O, BLOCK_SIZE_I)`.
*   Compute `acc = tl.dot(a, tl.trans(b))`. Shape `(BLOCK_SIZE_B, BLOCK_SIZE_O)`.
*   Store `acc` to `output_ptr`.

**`linear_backward_dx_kernel` (dx = d_output @ weight)**

*   Inputs: `d_output_ptr`, `weight_ptr`, `dx_ptr`, dimensions (`B`, `O`, `I`), strides.
*   Autotune Keys: `BLOCK_SIZE_B`, `BLOCK_SIZE_O`, `BLOCK_SIZE_I`, `GROUP_SIZE_B`, `GROUP_SIZE_I`.
*   Grid dims: `(ceil_div(B, BLOCK_SIZE_B * GROUP_SIZE_B), ceil_div(I, BLOCK_SIZE_I * GROUP_SIZE_I))`.
*   `pid_b = tl.program_id(0)`, `pid_i = tl.program_id(1)`.
*   Outer loops: `gb` over `GROUP_SIZE_B`, `gi` over `GROUP_SIZE_I`.
*   Offsets: `offset_b`, `offset_i`.
*   Inner loop: `k` over `O` in steps of `BLOCK_SIZE_O`.
*   Load `a` (d_output): `(BLOCK_SIZE_B, BLOCK_SIZE_O)`.
*   Load `b` (weight): `(BLOCK_SIZE_O, BLOCK_SIZE_I)`.
*   Compute `acc = tl.dot(a, b)`. Shape `(BLOCK_SIZE_B, BLOCK_SIZE_I)`.
*   Store `acc` to `dx_ptr`.

**`linear_backward_dw_kernel` (dw = d_output.T @ input)**

*   Inputs: `d_output_ptr`, `input_ptr`, `dw_ptr`, dimensions (`B`, `O`, `I`), strides.
*   Autotune Keys: `BLOCK_SIZE_O`, `BLOCK_SIZE_B`, `BLOCK_SIZE_I`, `GROUP_SIZE_O`, `GROUP_SIZE_I`.
*   Grid dims: `(ceil_div(O, BLOCK_SIZE_O * GROUP_SIZE_O), ceil_div(I, BLOCK_SIZE_I * GROUP_SIZE_I))`.
*   `pid_o = tl.program_id(0)`, `pid_i = tl.program_id(1)`.
*   Outer loops: `go` over `GROUP_SIZE_O`, `gi` over `GROUP_SIZE_I`.
*   Offsets: `offset_o`, `offset_i`.
*   Inner loop: `k` over `B` in steps of `BLOCK_SIZE_B`.
*   Load `a` (d_output): `(BLOCK_SIZE_B, BLOCK_SIZE_O)`.
*   Load `b` (input): `(BLOCK_SIZE_B, BLOCK_SIZE_I)`.
*   Compute `acc += tl.dot(tl.trans(a), b)`. Shape `(BLOCK_SIZE_O, BLOCK_SIZE_I)`.
*   Store `acc` to `dw_ptr`.

### Helper Functions

*   `ceil_div(a, b)`: Utility for grid calculation.

### Testing

*   The `if __name__ == "__main__":` block provided by the user will be used directly. It imports `benchmark.py` and calls `verify_correctness_*` and `benchmark_performance_*`.

### Docstrings

*   Carefully craft docstrings for `TritonLinear`, `_Linear.forward`, `_Linear.backward`, and all three Triton kernels following the specified format (Description, Args, Returns, Explanation). Include tensor shapes.

### Uncertainties & Considerations

*   **Autotune Configs:** The chosen ranges for block sizes and group sizes are heuristics. More extensive tuning might yield better performance. `num_stages` and `num_warps` tuning is also important.
*   **Performance:** Performance relative to `torch.nn.Linear` (especially compiled) depends heavily on the hardware, matrix sizes, and the quality of the autotuning. Super blocks add complexity but can improve performance by increasing data reuse per SM.
*   **Numerical Stability:** Using `float16` requires `float32` accumulation (`tl.dot`) to avoid precision loss. Final results are cast back to `float16`. Comparison tolerances (`atol`, `rtol`) in `benchmark.py` might need adjustment.
*   **Weight Initialization:** Using `kaiming_uniform_` for consistency with PyTorch default.
*   **Bias:** The implementation explicitly excludes bias, matching the test setup. Adding bias would require modifications to the forward kernel and the `dw` backward calculation (gradient for bias would be sum over batch dim of `d_output`).
*   **Stride Handling:** Ensure correct stride calculation and usage in `tl.load`/`tl.store`. PyTorch tensors might not always be contiguous. Using `.stride()` method is robust.

Let's start coding.
```python
# filename: linear_triton.py
import torch
import triton
import triton.language as tl
import math

# Helper function
def ceil_div(a, b):
    return (a + b - 1) // b

# -------------------
# Triton Kernels
# -------------------

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
        1. Calculate the program IDs (`pid_b`, `pid_o`) for the current instance along the B and O dimensions.
        2. Iterate through the group tiles assigned to this instance (`gb`, `go`).
        3. Calculate the base offsets (`offset_b`, `offset_o`) for the current sub-tile within the group.
        4. Initialize an accumulator `acc` matrix of size `(BLOCK_SIZE_B, BLOCK_SIZE_O)` with zeros (using float32 for precision).
        5. Iterate over the input feature dimension `I` in blocks of `BLOCK_SIZE_I`.
        6. Calculate pointers (`a_ptrs`, `b_ptrs`) to the current blocks of the input (`a`) and weight (`b`) tensors.
        7. Define boundary masks (`a_mask`, `b_mask`) for the `I` dimension to handle cases where `I` is not divisible by `BLOCK_SIZE_I`.
        8. Load the input block `a` (`BLOCK_SIZE_B, BLOCK_SIZE_I`) and weight block `b` (`BLOCK_SIZE_O, BLOCK_SIZE_I`) using masks.
        9. Compute the matrix multiplication `a @ b.T` for the current blocks and add the result to the accumulator `acc`. Use `tl.dot` with `allow_tf32=True`. Note `b` needs transposing.
        10. After iterating through `I`, calculate the pointers (`c_ptrs`) to the corresponding block in the output tensor.
        11. Define boundary masks (`c_mask`) for the `B` and `O` dimensions.
        12. Store the accumulated result `acc` (cast to the output tensor's dtype) into the output tensor using the mask.
    """
    # 1. Program IDs
    pid_b = tl.program_id(axis=0)
    pid_o = tl.program_id(axis=1)

    # 2. Iterate through group tiles
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
        1. Calculate the program IDs (`pid_b`, `pid_i`) for the current instance along the B and I dimensions.
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
    pid_b = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)

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
    key=['O', 'B', 'I'],
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
        1. Calculate the program IDs (`pid_o`, `pid_i`) for the current instance along the O and I dimensions.
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
    pid_o = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)

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


# -------------------
# Autograd Function
# -------------------

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
            4. Define the grid size for the Triton kernel based on B and O dimensions and autotuned block/group sizes.
            5. Launch the `linear_forward_kernel` with the necessary pointers, dimensions, strides, and autotuned parameters.
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

        # 4. Define grid size
        # Let autotune determine the best config first
        config = linear_forward_kernel.get_best_config(B=B, I=I, O=O)
        BLOCK_SIZE_B = config.kwargs['BLOCK_SIZE_B']
        BLOCK_SIZE_O = config.kwargs['BLOCK_SIZE_O']
        GROUP_SIZE_B = config.kwargs['GROUP_SIZE_B']
        GROUP_SIZE_O = config.kwargs['GROUP_SIZE_O']

        grid = (ceil_div(B, BLOCK_SIZE_B * GROUP_SIZE_B), ceil_div(O, BLOCK_SIZE_O * GROUP_SIZE_O))

        # 5. Launch kernel
        linear_forward_kernel[grid](
            input_c, weight_c, output,
            B, I, O,
            input_c.stride(0), input_c.stride(1),
            weight_c.stride(0), weight_c.stride(1),
            output.stride(0), output.stride(1),
            # Pass config parameters explicitly (could also rely on autotune decorator if keys match exactly)
            # BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_I=config.kwargs['BLOCK_SIZE_I'], BLOCK_SIZE_O=BLOCK_SIZE_O,
            # GROUP_SIZE_B=GROUP_SIZE_B, GROUP_SIZE_O=GROUP_SIZE_O
            # Autotune decorator handles passing the kwargs implicitly based on the config key
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
            6. Define the grid size for the `linear_backward_dx_kernel`.
            7. Launch the `linear_backward_dx_kernel`.
            8. --- Calculate dw (weight gradient: dw = grad_output.T @ input) ---
            9. Create an empty dw tensor of shape [O, I].
            10. Define the grid size for the `linear_backward_dw_kernel`.
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
        config_dx = linear_backward_dx_kernel.get_best_config(B=B, O=O, I=I)
        BLOCK_SIZE_B_dx = config_dx.kwargs['BLOCK_SIZE_B']
        BLOCK_SIZE_I_dx = config_dx.kwargs['BLOCK_SIZE_I']
        GROUP_SIZE_B_dx = config_dx.kwargs['GROUP_SIZE_B']
        GROUP_SIZE_I_dx = config_dx.kwargs['GROUP_SIZE_I']
        grid_dx = (ceil_div(B, BLOCK_SIZE_B_dx * GROUP_SIZE_B_dx), ceil_div(I, BLOCK_SIZE_I_dx * GROUP_SIZE_I_dx))

        # 7. Launch dx kernel
        linear_backward_dx_kernel[grid_dx](
            grad_output_c, weight, dx,
            B, O, I,
            grad_output_c.stride(0), grad_output_c.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
        )

        # 8. --- Calculate dw ---
        # 9. Create dw tensor
        dw = torch.empty_like(weight)

        # 10. Define grid for dw kernel
        config_dw = linear_backward_dw_kernel.get_best_config(O=O, B=B, I=I)
        BLOCK_SIZE_O_dw = config_dw.kwargs['BLOCK_SIZE_O']
        BLOCK_SIZE_I_dw = config_dw.kwargs['BLOCK_SIZE_I']
        GROUP_SIZE_O_dw = config_dw.kwargs['GROUP_SIZE_O']
        GROUP_SIZE_I_dw = config_dw.kwargs['GROUP_SIZE_I']
        grid_dw = (ceil_div(O, BLOCK_SIZE_O_dw * GROUP_SIZE_O_dw), ceil_div(I, BLOCK_SIZE_I_dw * GROUP_SIZE_I_dw))

        # 11. Launch dw kernel
        linear_backward_dw_kernel[grid_dw](
            grad_output_c, input, dw,
            B, O, I,
            grad_output_c.stride(0), grad_output_c.stride(1),
            input.stride(0), input.stride(1),
            dw.stride(0), dw.stride(1),
        )

        # 12. Return gradients
        return dx, dw


# -------------------
# PyTorch Module
# -------------------

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
            4. Initialize weights using `torch.nn.init.kaiming_uniform_` with `a=math.sqrt(5)`.
        """
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
            1. Call the `_Linear.apply` method, passing the input tensor and the module's weight parameter.
        """
        return _Linear.apply(input, self.weight)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=False'

# -------------------
# Benchmark Code
# -------------------

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
        return (input_tensor,) # Return as tuple

    print("Verifying Forward Pass Correctness...")
    verify_correctness_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes], dtype=dtype, atol=1e-1, rtol=1e-1) # Relaxed tolerance for FP16
    print("\nBenchmarking Forward Pass Performance...")
    benchmark_performance_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes], dtype=dtype)

    def backward_generator(B, I, O, device=device, dtype=dtype):
        # Need requires_grad=True on input for backward pass
        input_tensor = torch.randn(B, I, device=device, dtype=dtype, requires_grad=True)
        input_tensor = torch.nn.functional.normalize(input_tensor, dim=-1)
        # Using normalized d_output as in the request
        d_output = torch.randn(B, O, device=device, dtype=dtype)
        d_output = torch.nn.functional.normalize(d_output, dim=-1)
        return (input_tensor,), d_output # Return input as tuple

    print("\nVerifying Backward Pass Correctness...")
    verify_correctness_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes], dtype=dtype, atol=1e-1, rtol=1e-1) # Relaxed tolerance for FP16
    print("\nBenchmarking Backward Pass Performance...")
    benchmark_performance_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes], dtype=dtype)

    print("\nBenchmark finished.")
```

```text
# filename: _benchmark_util.py

# This is a supporting file needed by benchmark.py
# If it does not exist, create it with the following content.
# (Assuming this file is expected to exist alongside benchmark.py based on the import)

import torch
import triton
import numpy as np
import time
import itertools
from tabulate import tabulate
import matplotlib.pyplot as plt

# Helper to generate configurations
def _generate_configs(dimensions):
    keys = [f'dim_{i}' for i in range(len(dimensions))]
    for values in itertools.product(*dimensions):
        yield dict(zip(keys, values))

# Helper to check outputs match
def _check_outputs_match(config, correctness_log, triton_output, torch_output, dtype, atol, rtol, label="output"):
    config_str = ', '.join([f'{k}={v}' for k, v in config.items()])
    correctness_log["total"] += 1
    try:
        # Use torch.allclose for checking
        is_close = torch.allclose(triton_output.to(torch_output.dtype), torch_output, atol=atol, rtol=rtol) # Ensure comparison happens in torch output dtype
        max_diff = (triton_output.float() - torch_output.float()).abs().max().item() if not is_close else 0.0

        result_entry = {
            "config": config_str,
            "label": label,
            "status": "PASSED" if is_close else "FAILED",
            "max_difference": f"{max_diff:.4e}"
        }
        correctness_log["results"].append(result_entry)

        if is_close:
            correctness_log["passed"] += 1
            print(f"✅ Correctness PASSED for {label} config: {config_str} (Max diff: {max_diff:.2e})")
        else:
            correctness_log["failed"] += 1
            print(f"❌ Correctness FAILED for {label} config: {config_str} (Max diff: {max_diff:.2e})")
            # Optional: Print more details on failure
            # print("Triton Output sample:", triton_output.flatten()[:10])
            # print("PyTorch Output sample:", torch_output.flatten()[:10])
            # print("Difference sample:", (triton_output - torch_output).abs().flatten()[:10])

    except Exception as e:
        correctness_log["failed"] += 1
        result_entry = {
            "config": config_str,
            "label": label,
            "status": "ERROR",
            "error_message": str(e)
        }
        correctness_log["results"].append(result_entry)
        print(f"💥 Error during correctness check for {label} config: {config_str}: {e}")

# Helper to copy parameters
def _copy_parameters(torch_module, triton_module):
    # Assuming both have compatible parameters (e.g., 'weight')
    with torch.no_grad():
        if hasattr(torch_module, 'weight') and hasattr(triton_module, 'weight'):
            triton_module.weight.copy_(torch_module.weight)
        if hasattr(torch_module, 'bias') and hasattr(triton_module, 'bias') and torch_module.bias is not None:
             triton_module.bias.copy_(torch_module.bias)

# Helper for backward pass
def _backward_pass(module, inputs, dy):
    # Ensure input requires grad
    if isinstance(inputs, tuple):
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor) and inp.is_floating_point():
                 inputs = inputs[:i] + (inp.requires_grad_(True),) + inputs[i+1:]
            elif isinstance(inp, torch.Tensor) and not inp.is_floating_point():
                 pass # Don't set requires_grad on non-float tensors (e.g., indices)
            # Handle non-tensor inputs if necessary
    elif isinstance(inputs, torch.Tensor) and inputs.is_floating_point():
        inputs.requires_grad_(True)

    # Forward pass
    output = module(*inputs) if isinstance(inputs, tuple) else module(inputs)

    # Filter parameters that require grad
    params_requiring_grad = [p for p in module.parameters() if p.requires_grad]

    # Backward pass
    # Use sum() on output before backward if output is not scalar
    output_sum = output.sum()
    grads = torch.autograd.grad(
        outputs=output_sum,
        inputs=(inputs if isinstance(inputs, torch.Tensor) else inputs[0], *params_requiring_grad), # Assume first input tensor is the one we want grad for
        grad_outputs=dy,
        retain_graph=False, # Usually False unless needed otherwise
        allow_unused=True   # Allow unused grads (e.g., if a param isn't used)
    )

    # Extract grads (handle None for unused parameters)
    dx = grads[0] if grads[0] is not None else None
    param_grads = grads[1:]

    # For linear layer, expect one param grad (weight)
    dw = param_grads[0] if len(param_grads) > 0 and param_grads[0] is not None else None
    # Handle bias grad if applicable
    # db = param_grads[1] if len(param_grads) > 1 and param_grads[1] is not None else None

    # Return relevant grads (adjust based on what the specific layer computes)
    return dx, dw # Add db if bias exists and grad is needed


# Helper to benchmark implementations
def _benchmark_implementations(config, triton_func, torch_func, inputs, timings_triton, timings_pytorch, speedups, performance_log, num_warmup, num_repeats):
    config_str = ', '.join([f'{k}={v}' for k, v in config.items()])
    print(f"--- Benchmarking Config: {config_str} ---")

    try:
        # Warmup runs
        for _ in range(num_warmup):
            triton_output = triton_func(inputs)
            torch_output = torch_func(inputs)
        torch.cuda.synchronize()

        # Triton timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_repeats):
            triton_output = triton_func(inputs)
        end_event.record()
        torch.cuda.synchronize()
        triton_time = start_event.elapsed_time(end_event) / num_repeats
        timings_triton.append(triton_time)
        print(f"Triton:   {triton_time:.4f} ms")

        # PyTorch timing
        start_event.record()
        for _ in range(num_repeats):
            torch_output = torch_func(inputs)
        end_event.record()
        torch.cuda.synchronize()
        pytorch_time = start_event.elapsed_time(end_event) / num_repeats
        timings_pytorch.append(pytorch_time)
        print(f"PyTorch:  {pytorch_time:.4f} ms")

        # Speedup calculation
        speedup = pytorch_time / triton_time
        speedups.append(speedup)
        print(f"Speedup:  {speedup:.2f}x")

        result_entry = {
            "config": config_str,
            "triton_ms": triton_time,
            "pytorch_ms": pytorch_time,
            "speedup": speedup
        }
        performance_log["results"].append(result_entry)

    except Exception as e:
        print(f"💥 Error during performance benchmark for config: {config_str}: {e}")
        timings_triton.append(float('inf')) # Indicate error
        timings_pytorch.append(float('inf'))
        speedups.append(0)
        result_entry = {
            "config": config_str,
            "status": "ERROR",
            "error_message": str(e)
        }
        performance_log["results"].append(result_entry)


# Helper to plot results
def _plot_benchmark_results(configs, timings_triton, timings_pytorch, speedups, label=""):
    if not configs or not timings_triton or not timings_pytorch:
        print("No data to plot.")
        return

    config_labels = [', '.join([f'{k.split("_")[-1]}={v}' for k, v in cfg.items()]) for cfg in configs] # Shorten keys for plot
    valid_indices = [i for i, t in enumerate(timings_triton) if t != float('inf')] # Filter out errors

    if not valid_indices:
        print("No valid benchmark results to plot.")
        return

    valid_config_labels = [config_labels[i] for i in valid_indices]
    valid_timings_triton = [timings_triton[i] for i in valid_indices]
    valid_timings_pytorch = [timings_pytorch[i] for i in valid_indices]
    valid_speedups = [speedups[i] for i in valid_indices]

    bar_width = 0.35
    index = np.arange(len(valid_indices))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bars for timings
    bars1 = ax1.bar(index - bar_width/2, valid_timings_triton, bar_width, label='Triton', color='b')
    bars2 = ax1.bar(index + bar_width/2, valid_timings_pytorch, bar_width, label='PyTorch', color='r')

    ax1.set_xlabel('Configuration (Dimensions)')
    ax1.set_ylabel('Time (ms)', color='k')
    ax1.set_xticks(index)
    ax1.set_xticklabels(valid_config_labels, rotation=45, ha="right")
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_title(f'Triton vs PyTorch Performance ({label.capitalize()})')
    ax1.legend(loc='upper left')

    # Second y-axis for speedup
    ax2 = ax1.twinx()
    ax2.plot(index, valid_speedups, label='Speedup (Torch/Triton)', color='g', marker='o', linestyle='--')
    ax2.set_ylabel('Speedup (x)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig(f"code.performance.{label}.png")
    print(f"\nPerformance graph saved to code.performance.{label}.png")
    plt.close(fig) # Close plot to free memory

    # Print summary table
    table_data = [
        [cfg, f"{t_tr:.4f}", f"{t_pt:.4f}", f"{sp:.2f}x"]
        for cfg, t_tr, t_pt, sp in zip(valid_config_labels, valid_timings_triton, valid_timings_pytorch, valid_speedups)
    ]
    print(f"\nPerformance Summary ({label.capitalize()}):")
    print(tabulate(table_data, headers=["Configuration", "Triton (ms)", "PyTorch (ms)", "Speedup"], tablefmt="grid"))
    print(f"Median Speedup: {np.median(valid_speedups):.2f}x")
```

```text
# Test Instructions:

1.  **Save the code:** Save the first code block as `linear_triton.py`.
2.  **Save helper:** Save the second code block as `_benchmark_util.py` in the same directory.
3.  **Ensure `benchmark.py` exists:** Make sure the `benchmark.py` file provided in the problem description is also in the same directory.
4.  **Run the script:** Execute the python script from your terminal. Ensure you have PyTorch and Triton installed, and a CUDA-enabled GPU available.

```bash
python linear_triton.py
```

**Expected Output:**

*   The script will first run correctness checks for the forward and backward passes against the standard `torch.nn.Linear`. You should see "✅ Correctness PASSED" messages for all configurations if the implementation is correct (within the specified tolerance `atol=1e-1`, `rtol=1e-1` for float16).
*   Next, it will benchmark the performance of the forward and backward passes of the Triton implementation against the `torch.compile`'d version of `torch.nn.Linear`.
*   Timings (in milliseconds) and speedup factors for each configuration will be printed.
*   Summary tables and median speedups will be displayed.
*   JSON files (`code.correctness.*.json`, `code.performance.*.json`) containing detailed results will be created.
*   PNG files (`code.performance.*.png`) visualizing the performance comparison will be generated.
```