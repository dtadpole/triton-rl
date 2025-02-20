import torch

import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    gamma_ptr,  # Pointer to the gamma (weight) tensor
    input_row_stride,  # Stride for accessing rows of the input
    input_col_stride, # Stride for accessing columns of the input
    gamma_stride,  # Stride for accessing elements of gamma
    n_cols,  # Number of columns (features)
    eps, # Small value to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets * input_col_stride
    gamma_ptrs = gamma_ptr + col_offsets * gamma_stride

    # Load input and gamma
    input_values = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0.0)
    gamma_values = tl.load(gamma_ptrs, mask=col_offsets < n_cols, other=1.0) # Default to 1.0 if out of bound to handle edge case.

    # Compute RMS
    squares = input_values * input_values
    mean_squares = tl.sum(squares, axis=0) / n_cols
    rms = tl.sqrt(mean_squares + eps)

    # Normalize
    normalized_values = input_values / rms

     # Apply gamma
    output_values = normalized_values * gamma_values

    # Store output
    output_ptrs = output_ptr + row_idx * input_row_stride + col_offsets * input_col_stride
    tl.store(output_ptrs, output_values, mask=col_offsets < n_cols)



def rmsnorm(input: torch.Tensor, gamma: torch.Tensor, eps: float):

    if input.ndim != 2:
      raise ValueError("Input must have 2 dimensions: (rows, features)")

    n_rows, n_cols = input.shape
    output = torch.empty_like(input)

    # Determine block size (must be a power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > 2048:
        BLOCK_SIZE = 2048
    if BLOCK_SIZE < 16:
      BLOCK_SIZE = 16  # minimum size

    # Launch the kernel
    grid = (n_rows,)  # One kernel launch per row
    rmsnorm_kernel[grid](
        input,
        output,
        gamma,
        input.stride(0),
        input.stride(1),
        gamma.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

import torch.nn.functional as F
import torch

def rmsnorm_torch(input: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    # input : [rows, dim]
    rms = torch.sqrt(torch.mean(input * input, dim=-1, keepdim=True) + eps)
    return (input / rms) * gamma

def test_rmsnorm(rows, cols, eps, dtype, device):

    # Create input tensor
    torch.manual_seed(0)
    input = torch.randn((rows, cols), dtype=dtype, device=device)
    gamma = torch.randn((cols,), dtype=dtype, device=device)

    # Triton implementation
    output_triton = rmsnorm(input.clone(), gamma.clone(), eps)

    # PyTorch implementation
    output_torch = rmsnorm_torch(input.clone(), gamma.clone(), eps)

    # Compare results
    atol = 1e-2 if dtype == torch.float16 else 1e-4
    rtol = 1e-2 if dtype == torch.float16 else 1e-4  # relaxed tolerance for fp16
    is_close = torch.allclose(output_triton, output_torch, rtol=rtol, atol=atol)

    if not is_close:
        print(f"Test failed for shape ({rows}, {cols}), dtype={dtype}")
        print("Triton output:", output_triton)
        print("PyTorch output:", output_torch)
        diff = torch.abs(output_triton - output_torch)
        print("Absolute difference:", diff)
        print("Max difference:", torch.max(diff))
        assert False, "Test Failed."

    print(f"Test passed for shape ({rows}, {cols}), dtype={dtype}")
    return is_close

# Test cases
test_cases = [
    (10, 32, 1e-5, torch.float32, 'cuda'),
    (128, 64, 1e-6, torch.float32, 'cuda'),
    (512, 128, 1e-5, torch.float32, 'cuda'),
    (1024, 256, 1e-6, torch.float32, 'cuda'),
    (2048, 512, 1e-5, torch.float32, 'cuda'),
    (4096, 1024, 1e-6, torch.float32, 'cuda'),
    (1, 1, 1e-05, torch.float32, 'cuda'),  # Edge case: 1x1
    (1, 10, 1e-5, torch.float32, 'cuda'), # edge case 1 row
    (10, 1, 1e-5, torch.float32, 'cuda'),  # Edge case: 1 column
    (10, 32, 1e-5, torch.float16, 'cuda'),  # float16 test
    (128, 64, 1e-6, torch.float16, 'cuda'),
    (512, 128, 1e-5, torch.float16, 'cuda'),
    (1024, 256, 1e-5, torch.float16, 'cuda'),
    (33, 33, 1e-5, torch.float32, 'cuda'),  # Non power of two
    (129, 65, 1e-5, torch.float32, 'cuda'),
    (100, 10, 1e-06, torch.float32, 'cuda'),  # Testing 1e-6
]

for rows, cols, eps, dtype, device in test_cases:
    test_rmsnorm(rows, cols, eps, dtype, device)
