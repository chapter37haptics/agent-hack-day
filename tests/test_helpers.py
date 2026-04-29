import pytest
from taste.profile import TasteProfile, CritiqueResult, CRITIQUE_DIMENSIONS
from loop import initial_prompt, format_critique


def make_profile(**kwargs):
    defaults = dict(
        principles=["prefers asymmetric compositions"],
        palette_tendency="warm, desaturated",
        compositional_style="off-axis framing",
        emotional_register="quiet melancholy",
        texture_feel="analog grain",
        avoid_list=["saturated colors", "centered subjects"],
    )
    return TasteProfile(**{**defaults, **kwargs})


def make_critique(**kwargs):
    defaults = dict(
        score=7.2,
        breakdown={k: 7.0 for k in CRITIQUE_DIMENSIONS},
        reasoning="palette needs more warmth",
        revised_prompt="a street in Tokyo, warmer tones",
    )
    return CritiqueResult(**{**defaults, **kwargs})


def test_initial_prompt_includes_principles():
    p = make_profile(principles=["prefers asymmetry", "warm tones only"])
    out = initial_prompt("a street in Tokyo", p)
    assert "prefers asymmetry" in out
    assert "warm tones only" in out


def test_initial_prompt_formats_avoid_list():
    p = make_profile(avoid_list=["neon", "centered subjects"])
    out = initial_prompt("a street", p)
    assert "neon" in out and "centered subjects" in out


def test_format_critique_rounds_score():
    c = make_critique(score=7.234)
    out = format_critique(0, c)
    assert "7.2" in out
    assert "7.234" not in out


def test_format_critique_iterates_critique_dimensions_order():
    c = make_critique()
    out = format_critique(0, c)
    positions = [out.index(k) for k in CRITIQUE_DIMENSIONS if k in out]
    assert positions == sorted(positions)


def test_taste_profile_rejects_empty_principles():
    with pytest.raises(Exception):
        TasteProfile(
            principles=[],
            palette_tendency="warm",
            compositional_style="off-axis",
            emotional_register="quiet",
            texture_feel="grain",
            avoid_list=["neon"],
        )


def test_critique_result_rejects_out_of_range_score():
    with pytest.raises(Exception):
        CritiqueResult(score=15.0, breakdown={}, reasoning="", revised_prompt="")
