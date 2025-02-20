# filename: test_correctness.py
import torch
from rmsnorm import RMSNorm, TorchRMSNorm

M = 128
N = 512
dtype = torch.float32
x = torch.randn(M, N, device='cuda', dtype=dtype)
triton_rmsnorm = RMSNorm(N, dtype=dtype).to(device='cuda')
torch_rmsnorm = TorchRMSNorm(N, dtype=dtype).to(device='cuda')

# Copy the weights to ensure identical initialization
triton_rmsnorm.weight.data.copy_(torch_rmsnorm.weight.data)

y_triton = triton_rmsnorm(x)
y_torch = torch_rmsnorm(x)

# Compare the outputs with a tolerance
torch.testing.assert_close(y_triton, y_torch, rtol=1e-03, atol=1e-03, check_dtype=False)
print("Triton and PyTorch RMSNorm outputs match!")

