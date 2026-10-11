"""Inspect AI scorers for activation steering and intervention evaluations."""

from inspect_ai.scorer import Score, Scorer, Target, Value, scorer
from inspect_ai.solver import TaskState


@scorer(metrics=["accuracy", "mean"])
def logit_recovery_scorer(
    clean_key: str = "clean_target_logit",
    corrupted_key: str = "corrupted_target_logit",
    patched_key: str = "patched_target_logit",
) -> Scorer:
    """Computes the Logit Recovery Percentage under activation patching/steering.

    Formula: Recovery % = (Patched - Corrupted) / (Clean - Corrupted)
    """

    async def score(state: TaskState, target: Target) -> Score:
        clean_logit = state.store.get(clean_key, 0.0)
        corrupted_logit = state.store.get(corrupted_key, 0.0)
        patched_logit = state.store.get(patched_key, 0.0)

        denom = clean_logit - corrupted_logit
        if abs(denom) < 1e-6:
            recovery = 0.0
        else:
            recovery = (patched_logit - corrupted_logit) / denom

        return Score(
            value=Value.NUMBER(float(recovery)),
            answer=f"{recovery * 100:.2f}%",
            explanation=(
                f"Clean: {clean_logit:.3f}, Corrupted: {corrupted_logit:.3f}, "
                f"Patched: {patched_logit:.3f} -> Recovery: {recovery * 100:.2f}%"
            ),
            metadata={
                "clean_logit": clean_logit,
                "corrupted_logit": corrupted_logit,
                "patched_logit": patched_logit,
                "recovery_ratio": recovery,
            },
        )

    return score


@scorer(metrics=["accuracy", "mean"])
def steering_target_match_scorer(
    target_keywords: list[str],
) -> Scorer:
    """Evaluates whether output text generated under steering matches expected target keywords."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion.lower() if state.output else ""
        matched = [kw for kw in target_keywords if kw.lower() in completion]

        success_rate = len(matched) / len(target_keywords) if target_keywords else 0.0

        return Score(
            value=Value.NUMBER(success_rate),
            answer=completion,
            explanation=f"Matched keywords ({len(matched)}/{len(target_keywords)}): {matched}",
            metadata={"matched_keywords": matched, "completion": completion},
        )

    return score
