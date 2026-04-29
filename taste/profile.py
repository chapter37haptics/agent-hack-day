from __future__ import annotations
from pydantic import BaseModel, field_validator

CRITIQUE_DIMENSIONS = [
    "palette_tendency",
    "compositional_style",
    "emotional_register",
    "texture_feel",
    "avoid_list",
]


class TasteProfile(BaseModel):
    principles: list[str]
    palette_tendency: str
    compositional_style: str
    emotional_register: str
    texture_feel: str
    avoid_list: list[str]

    @field_validator("principles")
    @classmethod
    def principles_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("principles cannot be empty")
        return v


class CritiqueResult(BaseModel):
    score: float
    breakdown: dict[str, float]
    reasoning: str
    revised_prompt: str

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v: float) -> float:
        if not 0 <= v <= 10:
            raise ValueError("score must be 0-10")
        return v
