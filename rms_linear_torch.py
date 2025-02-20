import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float): A small value added to the denominator for numerical stability.
            Defaults to 1e-6.

    Attributes:
        weight (nn.Parameter): Learnable scaling parameter.
        eps (float): Small value for numerical stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RMSNorm layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x



class RMSNormLinear(nn.Module):
    """
    A module combining RMSNorm followed by a Linear layer.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the Linear layer will not learn an additive bias.
            Defaults to True.
        eps (float): A small value added to the denominator for numerical stability in RMSNorm.
            Defaults to 1e-6.

    Attributes:
        rmsnorm (RMSNorm): The RMSNorm layer.
        linear (nn.Linear): The Linear layer.

    Example:
        >>> model = RMSNormLinear(512, 1024)
        >>> input_tensor = torch.randn(32, 512)
        >>> output_tensor = model(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 1024])

        >>> # No bias in the linear layer
        >>> model_no_bias = RMSNormLinear(512, 1024, bias=False)
        >>> output_tensor_no_bias = model_no_bias(input_tensor)
        >>> output_tensor_no_bias.shape
        torch.Size([32, 1024])

    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, eps: float = 1e-6):
        super().__init__()
        self.rmsnorm = RMSNorm(in_features, eps=eps)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the combined RMSNorm and Linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after RMSNorm and Linear transformation.
        """
        x = self.rmsnorm(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    # Example Usage and Testing

    # Create a sample input tensor
    batch_size = 32
    seq_len = 10
    in_features = 512
    out_features = 1024
    x = torch.randn(batch_size, seq_len, in_features)

    # 1. RMSNorm layer
    rmsnorm = RMSNorm(in_features)
    normalized_x = rmsnorm(x)
    print(f"RMSNorm output shape: {normalized_x.shape}")  # Expected: torch.Size([32, 10, 512])
    # Check if weights are trainable
    print(f"RMSNorm weight requires grad: {rmsnorm.weight.requires_grad}")  # Expected: True


    # 2. Linear layer
    linear = nn.Linear(in_features, out_features)
    linear_x = linear(x)
    print(f"Linear output shape: {linear_x.shape}")  # Expected: torch.Size([32, 10, 1024])


    # 3. Combined RMSNormLinear
    rmsnorm_linear = RMSNormLinear(in_features, out_features)
    output = rmsnorm_linear(x)
    print(f"RMSNormLinear output shape: {output.shape}")  # Expected: torch.Size([32, 10, 1024])

    # 4. Test with different batch size and sequence length
    x_diff = torch.randn(64, 20, in_features)  # Different batch size and sequence length
    output_diff = rmsnorm_linear(x_diff)
    print(f"RMSNormLinear output shape (different input shape): {output_diff.shape}")  # Expected: torch.Size([64, 20, 1024])

    # 5. Test without bias in the linear layer
    rmsnorm_linear_no_bias = RMSNormLinear(in_features, out_features, bias=False)
    output_no_bias = rmsnorm_linear_no_bias(x)
    print(f"RMSNormLinear (no bias) output shape: {output_no_bias.shape}")  # Expected: torch.Size([32, 10, 1024])

    # 6. Test with a different eps value
    rmsnorm_linear_diff_eps = RMSNormLinear(in_features, out_features, eps=1e-8)
    output_diff_eps = rmsnorm_linear_diff_eps(x)
    print(f"RMSNormLinear (different eps) output shape: {output_diff_eps.shape}")  # Expected: torch.Size([32, 10, 1024])

    # 7. Test with a 2D input (no sequence length)
    x_2d = torch.randn(batch_size, in_features)
    output_2d = rmsnorm_linear(x_2d)
    print(f"RMSNormLinear output shape (2D input): {output_2d.shape}") # Expected: torch.Size([32, 1024])

    # 8. Check gradient flow
    loss = output.sum()
    loss.backward()
    print(f"RMSNorm weight gradient: {rmsnorm_linear.rmsnorm.weight.grad is not None}") # Expected: True
    print(f"Linear weight gradient: {rmsnorm_linear.linear.weight.grad is not None}")  # Expected: True
    if rmsnorm_linear.linear.bias is not None: # check for bias
        print(f"Linear bias gradient: {rmsnorm_linear.linear.bias.grad is not None}")  # Expected: True (if bias=True)
    else:
        print("Linear bias is None (bias=False)")
