import torch


def patch_hf_fp8_linear():
    """
    Make transformers' FP8Linear (used by *-FP8 checkpoints such as Qwen/Qwen3-14B-FP8)
    safe for CUDA-graph capture.

    The stock forward ends with torch.cuda.synchronize(), which aborts graph capture
    (cudaErrorStreamCaptureUnsupported) and adds a CPU sync per linear layer. The math is
    unchanged: dynamic per-block activation quant + Triton block-FP8 matmul.

    On DGX Spark (GB10, sm_121) the Triton kernels also need
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas, since Triton's bundled ptxas does not
    know sm_121a. Requires Python < 3.14 (torch.compile and Triton's frontend).
    """
    from transformers.integrations import finegrained_fp8 as fp8

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight.element_size() > 1:
            return torch.nn.functional.linear(input, self.weight, self.bias)
        qinput, scale = fp8.act_quant(input, self.block_size[1])
        output = fp8.w8a8_block_fp8_matmul_triton(
            qinput,
            self.weight,
            scale,
            self.weight_scale_inv,
            self.block_size,
            output_dtype=input.dtype,
        )
        if self.bias is not None:
            output = output + self.bias
        return output

    fp8.FP8Linear.forward = forward
