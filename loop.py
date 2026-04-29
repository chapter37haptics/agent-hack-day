from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from taste.profile import TasteProfile, CritiqueResult, CRITIQUE_DIMENSIONS

CONVERGENCE_THRESHOLD = 8.0
HUMAN_GATE_LOW = 6.0
MAX_ITERATIONS = 5


@dataclass
class LoopResult:
    final_image: bytes
    final_score: float
    iterations: int
    converged: bool


def initial_prompt(user_intent: str, profile: TasteProfile) -> str:
    principles_str = " ".join(profile.principles)
    return (
        f"{user_intent}. "
        f"Taste principles: {principles_str}. "
        f"Palette: {profile.palette_tendency}. "
        f"Composition: {profile.compositional_style}. "
        f"Mood: {profile.emotional_register}. "
        f"Texture: {profile.texture_feel}. "
        f"Avoid: {', '.join(profile.avoid_list)}."
    )


def format_critique(i: int, result: CritiqueResult) -> str:
    dims = " | ".join(
        f"{k}: {result.breakdown.get(k, 0):.1f}" for k in CRITIQUE_DIMENSIONS
    )
    return (
        f"**Iteration {i + 1}** — score {result.score:.1f}/10\n"
        f"{dims}\n"
        f"_{result.reasoning}_\n"
        f"→ {result.revised_prompt}"
    )


def run(
    prompt: str,
    taste_profile: TasteProfile,
    generator,
    critic,
    on_iteration_complete: Callable[[int, bytes, CritiqueResult], None],
    on_human_gate: Callable[[bytes, CritiqueResult], bool],
    max_iterations: int = MAX_ITERATIONS,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    human_gate_low: float = HUMAN_GATE_LOW,
) -> LoopResult:
    best_image: bytes | None = None
    best_score = 0.0

    for i in range(max_iterations):
        try:
            image = generator.generate(prompt)
        except Exception as e:
            if best_image is not None:
                return LoopResult(best_image, best_score, i, False)
            raise

        try:
            result = critic.evaluate(image, taste_profile)
        except Exception:
            try:
                result = critic.evaluate(image, taste_profile)
            except Exception:
                continue

        if result.score > best_score:
            best_image, best_score = image, result.score

        on_iteration_complete(i, image, result)

        if result.score >= convergence_threshold:
            return LoopResult(image, result.score, i + 1, True)

        if result.score >= human_gate_low:
            approved = on_human_gate(image, result)
            if approved:
                return LoopResult(image, result.score, i + 1, False)

        prompt = result.revised_prompt

    return LoopResult(best_image or b"", best_score, max_iterations, False)
