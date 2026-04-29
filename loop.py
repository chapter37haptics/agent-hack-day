from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Protocol
from taste.profile import TasteProfile, CritiqueResult, CRITIQUE_DIMENSIONS

CONVERGENCE_THRESHOLD = 8.0
HUMAN_GATE_LOW = 6.0
MAX_ITERATIONS = 5


class GeneratorProtocol(Protocol):
    def generate(self, prompt: str) -> bytes: ...


class CriticProtocol(Protocol):
    def evaluate(self, image: bytes, profile: TasteProfile) -> CritiqueResult: ...


@dataclass
class LoopResult:
    final_image: bytes
    final_score: float
    iterations: int
    converged: bool
    human_approved: bool = field(default=False)


def initial_prompt(user_intent: str, profile: TasteProfile) -> str:
    principles_str = "; ".join(profile.principles)
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
    generator: GeneratorProtocol,
    critic: CriticProtocol,
    on_iteration_complete: Callable[[int, bytes, CritiqueResult], None],
    on_human_gate: Callable[[bytes, CritiqueResult], bool],
    max_iterations: int = MAX_ITERATIONS,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    human_gate_low: float = HUMAN_GATE_LOW,
) -> LoopResult:
    if human_gate_low >= convergence_threshold:
        raise ValueError(
            f"human_gate_low ({human_gate_low}) must be < convergence_threshold ({convergence_threshold})"
        )

    best_image: bytes = b""
    best_score: float = 0.0

    for i in range(max_iterations):
        try:
            image = generator.generate(prompt)
        except Exception as e:
            if best_image:
                return LoopResult(best_image, best_score, i, converged=False)
            raise RuntimeError(f"Generation failed on iteration {i + 1} with no prior result: {e}") from e

        # Critic: single attempt; caller controls retry policy via on_iteration_complete
        try:
            result = critic.evaluate(image, taste_profile)
        except Exception as e:
            # Skip iteration — use unchanged prompt
            on_iteration_complete(i, image, CritiqueResult(
                score=0.0,
                breakdown={k: 0.0 for k in CRITIQUE_DIMENSIONS},
                reasoning=f"Critic failed: {e}",
                revised_prompt=prompt,
            ))
            continue

        if result.score > best_score:
            best_image, best_score = image, result.score

        on_iteration_complete(i, image, result)

        if result.score >= convergence_threshold:
            return LoopResult(image, result.score, i + 1, converged=True)

        if result.score >= human_gate_low:
            approved = on_human_gate(image, result)
            if approved:
                return LoopResult(image, result.score, i + 1, converged=False, human_approved=True)

        prompt = result.revised_prompt

    return LoopResult(best_image, best_score, max_iterations, converged=False)
