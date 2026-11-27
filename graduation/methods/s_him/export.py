"""Verify CaRe-BN fusion before exporting the deployment policy."""

import torch


@torch.no_grad()
def check_and_export(model, observations, path):
    """Export a verified fused policy or preserve the original normalization."""
    previous_precision = torch.get_float32_matmul_precision()
    previous_conv_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        fused = model.fused()
        expected = model(observations)
        error = float((fused(observations) - expected).abs().max())
        result = {"fusion_max_error": error, "fusion_verified": error <= 1e-5}
        if path is None:
            return result
        deploy = fused if result["fusion_verified"] else model.eval()
        compiled = torch.jit.freeze(torch.jit.trace(deploy, observations, check_trace=False))
        torch.testing.assert_close(compiled(observations), expected, rtol=1e-5, atol=1e-5)
        compiled.save(str(path))
        result["deployment"] = "fused" if result["fusion_verified"] else "unfused"
        return result
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cudnn.allow_tf32 = previous_conv_tf32
