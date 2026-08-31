"""PyTorch #188890: compiled BF16 gradients differ between identical runs."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as functional

from accel_verify import ComparisonConfig


if not torch.cuda.is_available():
    raise RuntimeError("pytorch_188890_cuda.py requires a CUDA-capable runner")

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = True


def forward(inputs, score_weight):
    length, dim = inputs.shape
    scores = torch.einsum("td,gd->gt", inputs, score_weight)
    score_delta = scores[:, None, :] - scores[:, :, None]

    pair_scores = torch.einsum("tn,Tn->tT", inputs, inputs)
    pair_weights = pair_scores[None] * torch.exp(score_delta)
    context = torch.einsum("gtT,Td->gtd", pair_weights, inputs)
    context = functional.layer_norm(context, (dim,))

    return context.repeat_interleave(4, dim=-1)[:1, :, :1].view(length, 1)


compiled = torch.compile(forward, backend="inductor")


def compiled_with_autocast(inputs, score_weight):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return compiled(inputs, score_weight)


torch.manual_seed(1337)
score_weight = (
    torch.empty(4, 224, device="cuda").normal_(std=0.02).requires_grad_()
)
model_inputs = torch.empty(296, 224, device="cuda").normal_(std=0.02)

# This issue is a self-consistency failure: two identical compiled executions
# should produce identical outputs and gradients in deterministic mode.
reference = compiled_with_autocast
candidate = compiled_with_autocast
inputs = (model_inputs, score_weight)
config = ComparisonConfig(
    rtol=0.0,
    atol=0.0,
    check_gradients=True,
    seed=1337,
)
metadata = {
    "source": "https://github.com/pytorch/pytorch/issues/188890",
    "surface": "compiled BF16 gradient self-consistency",
    "reported_reproducing_gpu": "NVIDIA H200",
    "reported_non_reproducing_gpu": "NVIDIA H100",
}


def _collect_gradient():
    score_weight.grad = None
    output = compiled_with_autocast(model_inputs, score_weight)
    output.sum().backward()
    return score_weight.grad.detach().float().reshape(-1).cpu()


def main():
    print(
        f"torch={torch.__version__} cuda_build={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name()} runs=2 failure_threshold=1.0"
    )
    grads_a = _collect_gradient()
    grads_b = _collect_gradient()
    grad_left_norm = float(torch.linalg.vector_norm(grads_a))
    grad_right_norm = float(torch.linalg.vector_norm(grads_b))
    grad_rel = float(torch.linalg.vector_norm(grads_b - grads_a)) / max(
        grad_left_norm, 1e-30
    )
    all_grads_finite = bool(torch.isfinite(grads_a).all()) and bool(
        torch.isfinite(grads_b).all()
    )
    failed = not all_grads_finite or grad_rel >= 1.0
    print(
        f"failed={failed} all_grads_finite={all_grads_finite} "
        f"max_seen_grad_rel={grad_rel:.8e} "
        f"grad0_norm={grad_left_norm:.8e} "
        f"grad1_norm={grad_right_norm:.8e}"
    )
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
