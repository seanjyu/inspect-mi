"""Inspect AI scorers for linear probing and representation analysis."""

import torch
from inspect_ai.scorer import Score, Scorer, Target, Value, scorer
from inspect_ai.solver import TaskState


@scorer(metrics=["accuracy", "mean"])
def probe_accuracy_scorer(
    buffer_key: str = "probe_outputs",
    threshold: float = 0.5,
) -> Scorer:
    """Inspect scorer that computes classification accuracy from probe outputs in context.buffer.

    Args:
        buffer_key: Key in sample/context buffer holding probe probability tensors.
        threshold: Decision boundary for binary classification.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Retrieve recorded probe outputs from task state metadata or buffer
        probe_outputs = state.store.get(buffer_key, [])
        if not probe_outputs:
            return Score(
                value=Value.NUMBER(0.0),
                explanation=f"No probe outputs found under key '{buffer_key}'",
            )

        # Concatenate step-wise probabilities across generation steps: [num_steps, ...]
        if isinstance(probe_outputs[0], torch.Tensor):
            probs = torch.stack(probe_outputs).squeeze()
        else:
            probs = torch.tensor(probe_outputs)

        # Ground truth target (e.g., binary label 0 or 1)
        ground_truth = float(target.text.strip())

        # Take the probe's prediction at the final position/step
        final_prob = probs[-1].item() if probs.dim() > 0 else probs.item()
        pred_label = 1.0 if final_prob >= threshold else 0.0

        is_correct = pred_label == ground_truth

        return Score(
            value=Value.NUMBER(1.0 if is_correct else 0.0),
            answer=f"prob={final_prob:.4f} (pred={int(pred_label)})",
            explanation=f"Ground Truth: {ground_truth}, Final Probe Prob: {final_prob:.4f}",
            metadata={
                "final_prob": final_prob,
                "all_probs": probs.tolist()
                if hasattr(probs, "tolist")
                else probe_outputs,
            },
        )

    return score
