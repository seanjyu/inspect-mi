import torch.nn.functional as F
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import Generate, TaskState, solver

from inspect_mi.core.context import MIContext, TargetSpec
from inspect_mi.core.hooks import apply_nnterp_hooks, bind_context


def _load_model():
    from nnterp import StandardizedTransformer  # or your fixture's constructor

    m = StandardizedTransformer("gpt2")
    m.eval()
    return m


@solver
def anti_steer_pass(scale: float = 50.0):
    model = _load_model()
    cfg = model.config
    n = getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 2))
    layer = max(0, n // 2)
    path = f"layers.{layer}.mlp"

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        text = state.input_text

        with model.trace(text):
            clean = model.output.logits[0, -1, :].save()
        clean = getattr(clean, "value", clean)
        top = clean.argmax().item()

        wte = getattr(model, "transformer", model).wte.weight
        vec = wte[top].detach().clone()
        vec -= vec.mean()
        vec /= vec.norm()

        ctx = MIContext(sample_id=str(state.sample_id))
        ctx.targets[path] = TargetSpec(
            layer=layer, location="mlp_out", module_path=path
        )
        ctx.mutators[path] = lambda act, s: act - scale * vec.to(act.device, act.dtype)

        with bind_context(ctx), model.trace(text):
            apply_nnterp_hooks(model, ctx)
            steered = model.output.logits[0, -1, :].save()
        steered = getattr(steered, "value", steered)

        cp, sp = F.softmax(clean, -1)[top].item(), F.softmax(steered, -1)[top].item()
        state.store.set("mi:max_logit_diff", (clean - steered).abs().max().item())
        state.store.set("mi:target_logit_drop", (clean[top] - steered[top]).item())
        state.store.set("mi:rel_prob_drop", (cp - sp) / (cp + 1e-8))
        state.store.set("mi:layer", layer)
        state.output.completion = f"top_token={top}"
        return state

    return solve


@scorer(metrics=[accuracy()])
def steering_smoke():
    async def score(state: TaskState, target: Target) -> Score:
        d = state.store.get("mi:max_logit_diff")
        r = state.store.get("mi:rel_prob_drop")
        checks = {
            "global_perturb": d > 0.5,
            "rel_prob": r > 0.30,
        }
        return Score(
            value=CORRECT if all(checks.values()) else INCORRECT,
            explanation=f"max_diff={d:.3f} rel={r:.1%} failed={[k for k, v in checks.items() if not v]}",
            metadata={**checks, "layer": state.store.get("mi:layer")},
        )

    return score


@task
def anti_steer_smoke(scale: float = 50.0):
    return Task(
        dataset=MemoryDataset(
            [
                Sample(
                    input="The quick brown fox jumps over the lazy dog",
                    id="e2e_anti_steer",
                )
            ]
        ),
        solver=anti_steer_pass(scale),
        scorer=steering_smoke(),
    )
