from __future__ import annotations
import base64
import json
from enum import Enum, auto
from pathlib import Path

import streamlit as st

from loop import (
    CONVERGENCE_THRESHOLD,
    HUMAN_GATE_LOW,
    MAX_ITERATIONS,
    initial_prompt,
)
from taste import critic as critic_module
from taste import extractor
from generator import client as generator_module
from taste.profile import TasteProfile, CRITIQUE_DIMENSIONS


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIR Generator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp { background: #0d0d0d; color: #e0e0e0; }
.stApp header { background: #0d0d0d; }

/* Step trail */
.step-trail {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 0 18px 0; font-family: monospace; font-size: 12px;
}
.step-done  { color: #4a8; }
.step-active { color: #c8a96e; font-weight: bold; }
.step-todo  { color: #444; }
.step-sep   { color: #333; }

/* Iteration cards */
.iter-card {
    background: #1a1a1a; border: 1px solid #252525;
    padding: 10px 12px; margin-bottom: 8px; border-radius: 2px;
}
.iter-card.active { border-color: #c8a96e; }
.iter-header { display: flex; justify-content: space-between; margin-bottom: 6px; }
.iter-num { font-family: monospace; font-size: 11px; color: #555; letter-spacing: 0.1em; }
.score-high { color: #4a8; font-size: 11px; }
.score-mid  { color: #ca6; font-size: 11px; }
.score-low  { color: #c87; font-size: 11px; }
.score-pending { color: #444; font-size: 11px; }
.iter-critique { font-size: 11px; color: #888; line-height: 1.5; margin-bottom: 5px; }
.iter-delta { font-size: 10px; color: #666; font-style: italic; }

/* Taste principles */
.principle { background: #1c1c1c; border-left: 2px solid #c8a96e;
    padding: 5px 10px; font-size: 12px; color: #bbb; margin-bottom: 4px; }
.principle.negative { border-left-color: #7a3a3a; color: #999; }

/* Score overlay card */
.score-display {
    background: #141414; border: 1px solid #1e1e1e;
    padding: 12px 16px; display: flex; justify-content: space-between;
    align-items: center; margin-top: 8px;
}
.score-big { font-size: 32px; font-weight: bold; color: #4a8; font-family: monospace; }
.score-label { font-size: 10px; color: #555; letter-spacing: 0.12em; text-transform: uppercase; }

/* Breakdown table */
.breakdown-row { display: flex; justify-content: space-between;
    font-size: 11px; padding: 4px 0; border-bottom: 1px solid #1a1a1a; }
.breakdown-key { color: #666; }
.breakdown-high { color: #4a8; }
.breakdown-mid  { color: #ca6; }
.breakdown-low  { color: #c87; }

/* Human gate */
.gate-box {
    background: #1e1a14; border: 1px solid #4a3a20;
    padding: 12px; margin-top: 8px;
}
.gate-question { font-size: 12px; color: #c8a96e; margin-bottom: 10px; font-family: monospace; }

/* Section label */
.section-label {
    font-size: 10px; color: #555; letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 8px; font-family: monospace;
}

/* Spinner inline */
@keyframes spin { to { transform: rotate(360deg); } }
.spinner-inline {
    display: inline-block; width: 8px; height: 8px;
    border: 1px solid #444; border-top-color: #c8a96e;
    border-radius: 50%; animation: spin 0.8s linear infinite;
    margin-left: 6px; vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)


# ── Phase state machine ────────────────────────────────────────────────────────
class Phase(Enum):
    UPLOAD = auto()
    CONFIRM = auto()
    GENERATING = auto()
    DONE = auto()


if "phase" not in st.session_state:
    st.session_state["phase"] = Phase.UPLOAD

STEP_LABELS = [
    ("UPLOAD", Phase.UPLOAD, "① EXTRACT TASTE"),
    ("CONFIRM", Phase.CONFIRM, "② CONFIRM"),
    ("DESCRIBE", None, "③ DESCRIBE"),
    ("GENERATING", Phase.GENERATING, "④ GENERATE + CRITIQUE"),
    ("DONE", Phase.DONE, "⑤ APPROVE"),
]

_PHASE_ORDER = [Phase.UPLOAD, Phase.CONFIRM, Phase.GENERATING, Phase.DONE]


def render_step_trail() -> None:
    phase = st.session_state["phase"]
    # Map phases to step indices: UPLOAD=0, CONFIRM=1, GENERATING=2+3, DONE=4
    current_step = {
        Phase.UPLOAD: 0,
        Phase.CONFIRM: 1,
        Phase.GENERATING: 3,
        Phase.DONE: 4,
    }.get(phase, 0)

    parts: list[str] = []
    for i, (_, _, label) in enumerate(STEP_LABELS):
        if i < current_step:
            parts.append(f'<span class="step-done">{label}</span>')
        elif i == current_step:
            parts.append(f'<span class="step-active">{label}</span>')
        else:
            parts.append(f'<span class="step-todo">{label}</span>')

        if i < len(STEP_LABELS) - 1:
            parts.append('<span class="step-sep">›</span>')

    st.markdown(
        f'<div class="step-trail"><span style="color:#c8a96e;font-family:monospace;font-weight:bold;margin-right:8px;">AIR</span>'
        + " ".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def _score_class(score: float) -> str:
    if score >= 8.0:
        return "score-high"
    elif score >= 5.0:
        return "score-mid"
    return "score-low"


def _score_symbol(score: float) -> str:
    if score >= 8.0:
        return "✓"
    elif score >= 5.0:
        return "~"
    return "✗"


def render_breakdown(result) -> None:
    """Render per-dimension score breakdown as color-coded rows."""
    rows = []
    for k in CRITIQUE_DIMENSIONS:
        v = result.breakdown.get(k, 0.0)
        sym = _score_symbol(v)
        css = "breakdown-" + ("high" if v >= 8 else "mid" if v >= 5 else "low")
        label = k.replace("_", " ")
        rows.append(
            f'<div class="breakdown-row">'
            f'<span class="breakdown-key">{label}</span>'
            f'<span class="{css}">{sym} {v:.0f}/10</span>'
            f"</div>"
        )
    st.markdown(
        '<div style="background:#141414;border:1px solid #1e1e1e;padding:14px;margin-top:8px;">'
        '<div class="section-label" style="margin-bottom:10px;">last critique breakdown</div>'
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


def _taste_md(profile: TasteProfile) -> str:
    lines = ["# Taste Profile\n"]
    lines.append("## Principles\n")
    for p in profile.principles:
        lines.append(f"- {p}")
    lines.append(f"\n**Palette:** {profile.palette_tendency}")
    lines.append(f"\n**Composition:** {profile.compositional_style}")
    lines.append(f"\n**Emotional register:** {profile.emotional_register}")
    lines.append(f"\n**Texture:** {profile.texture_feel}")
    lines.append(f"\n**Avoid:** {', '.join(profile.avoid_list)}")
    return "\n".join(lines)


# ── Human gate (checked before any phase render) ──────────────────────────────
if "pending_gate" in st.session_state:
    gate = st.session_state["pending_gate"]
    render_step_trail()

    left, right = st.columns([1, 1])

    with right:
        st.image(gate["image"], use_container_width=True)
        score = gate["result"].score
        st.markdown(
            f'<div class="score-display">'
            f'<div><div class="score-big {_score_class(score)}">{score:.1f}</div>'
            f'<div class="score-label">taste match score</div></div>'
            f'<div style="text-align:right;font-family:monospace;font-size:11px;color:#555;">'
            f'iteration {gate["iteration"] + 1} of max {MAX_ITERATIONS}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        render_breakdown(gate["result"])

    with left:
        profile: TasteProfile = st.session_state.get("taste_profile")
        if profile:
            st.markdown('<div class="section-label">extracted taste profile</div>', unsafe_allow_html=True)
            for p in profile.principles:
                st.markdown(f'<div class="principle">{p}</div>', unsafe_allow_html=True)
            avoid = "; ".join(profile.avoid_list)
            st.markdown(f'<div class="principle negative">AVOID: {avoid}</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="gate-box"><div class="gate-question">'
            f'critic is uncertain (score: {gate["result"].score:.1f}). does this feel like your taste?'
            f"</div></div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        if c1.button("yes — approve", use_container_width=True, type="primary"):
            st.session_state["final_image"] = gate["image"]
            st.session_state["final_result"] = gate["result"]
            del st.session_state["pending_gate"]
            st.session_state["phase"] = Phase.DONE
            st.rerun()
        if c2.button("almost — keep iterating", use_container_width=True):
            st.session_state["resume_prompt"] = gate["result"].revised_prompt
            del st.session_state["pending_gate"]
            st.rerun()
        if c3.button("no — restart", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.stop()


# ── Phase renderers ────────────────────────────────────────────────────────────

def render_upload() -> None:
    render_step_trail()
    st.markdown("Upload 3–5 reference images that represent your aesthetic taste.")

    files = st.file_uploader(
        "Reference images",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
    )

    if files and len(files) < 3:
        st.warning("Upload at least 3 images.")
        return

    if files and len(files) >= 3:
        cols = st.columns(min(len(files), 5))
        for col, f in zip(cols, files):
            col.image(f, use_container_width=True)

        if st.button("Extract taste →", type="primary", use_container_width=True):
            with st.spinner("Analyzing your aesthetic taste..."):
                tmp_paths: list[Path] = []
                for f in files:
                    tmp = Path(f"/tmp/air_ref_{f.name}")
                    tmp.write_bytes(f.read())
                    tmp_paths.append(tmp)
                try:
                    profile = extractor.extract(tmp_paths)
                    st.session_state["taste_profile"] = profile
                    st.session_state["ref_images"] = [f.read() if hasattr(f, 'read') else None for f in files]
                    st.session_state["phase"] = Phase.CONFIRM
                    st.rerun()
                except Exception as e:
                    st.error(f"Extraction failed: {e}")


def render_confirm() -> None:
    render_step_trail()
    profile: TasteProfile = st.session_state["taste_profile"]

    st.markdown("Edit any principles that feel off, then confirm.")

    st.markdown('<div class="section-label">overarching principles</div>', unsafe_allow_html=True)
    for i, p in enumerate(profile.principles):
        st.text_area(f"Principle {i + 1}", value=p, key=f"principle_{i}", height=60, label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">palette tendency</div>', unsafe_allow_html=True)
        st.text_input("palette", value=profile.palette_tendency, key="palette_tendency", label_visibility="collapsed")
        st.markdown('<div class="section-label">compositional style</div>', unsafe_allow_html=True)
        st.text_input("composition", value=profile.compositional_style, key="compositional_style", label_visibility="collapsed")
        st.markdown('<div class="section-label">emotional register</div>', unsafe_allow_html=True)
        st.text_input("emotional register", value=profile.emotional_register, key="emotional_register", label_visibility="collapsed")
    with col2:
        st.markdown('<div class="section-label">texture feel</div>', unsafe_allow_html=True)
        st.text_input("texture", value=profile.texture_feel, key="texture_feel", label_visibility="collapsed")
        st.markdown('<div class="section-label">avoid list (comma-separated)</div>', unsafe_allow_html=True)
        st.text_area("avoid", value=", ".join(profile.avoid_list), key="avoid_list_text", height=80, label_visibility="collapsed")

    st.markdown('<div class="section-label" style="margin-top:16px;">generation prompt</div>', unsafe_allow_html=True)
    user_intent = st.text_input(
        "Describe the subject",
        placeholder="e.g. a street in Tokyo at dusk, empty",
        key="user_intent",
        label_visibility="collapsed",
    )

    if st.button("Confirm taste & generate →", type="primary", use_container_width=True):
        if not user_intent.strip():
            st.warning("Please describe what you want to generate.")
            return

        confirmed_principles = [
            st.session_state[f"principle_{i}"]
            for i in range(len(profile.principles))
        ]
        updated = profile.model_copy(update={
            "principles": [p for p in confirmed_principles if p.strip()],
            "palette_tendency": st.session_state["palette_tendency"],
            "compositional_style": st.session_state["compositional_style"],
            "emotional_register": st.session_state["emotional_register"],
            "texture_feel": st.session_state["texture_feel"],
            "avoid_list": [
                s.strip()
                for s in st.session_state["avoid_list_text"].split(",")
                if s.strip()
            ],
        })
        st.session_state["taste_profile"] = updated
        st.session_state["user_intent"] = user_intent.strip()
        st.session_state["phase"] = Phase.GENERATING
        st.session_state["best_image"] = None
        st.session_state["best_score"] = 0.0
        st.rerun()


def _iter_card_html(i: int, score: float | None, critique: str, delta: str, active: bool = False) -> str:
    active_class = " active" if active else ""
    if score is None:
        score_html = f'<span class="score-pending">evaluating... <span class="spinner-inline"></span></span>'
        iter_html = f'<span class="iter-critique" style="color:#444">{critique}</span>'
    else:
        sc = _score_class(score)
        score_html = f'<span class="{sc}">taste match: {score:.1f} / 10</span>'
        iter_html = f'<span class="iter-critique">{critique}</span>'
        if delta:
            iter_html += f'<div class="iter-delta">→ {delta}</div>'

    return (
        f'<div class="iter-card{active_class}">'
        f'<div class="iter-header">'
        f'<span class="iter-num">ITERATION {i + 1}</span>'
        f"{score_html}"
        f"</div>"
        f"{iter_html}"
        f"</div>"
    )


def render_generating() -> None:
    render_step_trail()
    profile: TasteProfile = st.session_state["taste_profile"]
    user_intent: str = st.session_state["user_intent"]

    left, right = st.columns([1, 1])

    with left:
        # Reference images row
        st.markdown('<div class="section-label">reference images</div>', unsafe_allow_html=True)
        ref_imgs = st.session_state.get("ref_images", [])
        if ref_imgs:
            rcols = st.columns(min(len(ref_imgs), 5))
            for col, img in zip(rcols, ref_imgs):
                if img:
                    col.image(img, use_container_width=True)

        # Taste profile
        st.markdown('<div class="section-label" style="margin-top:12px;">extracted taste profile · confirmed ✓</div>', unsafe_allow_html=True)
        for p in profile.principles:
            st.markdown(f'<div class="principle">{p}</div>', unsafe_allow_html=True)
        avoid = "; ".join(profile.avoid_list)
        st.markdown(f'<div class="principle negative">AVOID: {avoid}</div>', unsafe_allow_html=True)

        # Generation prompt
        st.markdown('<div class="section-label" style="margin-top:12px;">generation prompt</div>', unsafe_allow_html=True)
        prompt_container = st.empty()
        prompt_container.caption(f"`{user_intent}`")

        # Critique iterations log
        st.markdown('<div class="section-label" style="margin-top:12px;">critique loop</div>', unsafe_allow_html=True)
        iter_containers = [st.empty() for _ in range(MAX_ITERATIONS)]

    with right:
        image_container = st.empty()
        score_container = st.empty()
        breakdown_container = st.empty()
        status_container = st.empty()

        # Placeholder before first image
        image_container.markdown(
            '<div style="background:#141414;aspect-ratio:1;display:flex;align-items:center;'
            'justify-content:center;color:#333;font-family:monospace;font-size:12px;">'
            'awaiting first generation...</div>',
            unsafe_allow_html=True,
        )

    prompt = st.session_state.pop("resume_prompt", None) or initial_prompt(user_intent, profile)

    for i in range(MAX_ITERATIONS):
        # Show "evaluating..." state in iteration card
        with left:
            iter_containers[i].markdown(
                _iter_card_html(i, None, "generating image...", "", active=True),
                unsafe_allow_html=True,
            )

        with right:
            status_container.caption(f"iteration {i + 1} of max {MAX_ITERATIONS} — generating...")

        with st.spinner(f""):
            try:
                image_bytes = generator_module.generate(prompt)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                best = st.session_state.get("best_image")
                if best:
                    st.session_state["final_image"] = best
                    st.session_state["phase"] = Phase.DONE
                    st.rerun()
                return

        with right:
            image_container.image(image_bytes, use_container_width=True)
            status_container.caption(f"iteration {i + 1} of max {MAX_ITERATIONS} — critiquing...")

        # Show "critiquing..." state in card
        with left:
            iter_containers[i].markdown(
                _iter_card_html(i, None, "critic is analyzing image against taste principles...", "", active=True),
                unsafe_allow_html=True,
            )

        try:
            result = critic_module.evaluate(image_bytes, profile)
        except Exception:
            try:
                result = critic_module.evaluate(image_bytes, profile)
            except Exception as e:
                with left:
                    iter_containers[i].markdown(
                        _iter_card_html(i, None, f"critic failed: {e}", "", active=False),
                        unsafe_allow_html=True,
                    )
                continue

        if result.score > st.session_state.get("best_score", 0.0):
            st.session_state["best_image"] = image_bytes
            st.session_state["best_score"] = result.score

        # Update iteration card with real result
        with left:
            iter_containers[i].markdown(
                _iter_card_html(
                    i,
                    result.score,
                    result.reasoning,
                    f"revised prompt: {result.revised_prompt[:80]}{'...' if len(result.revised_prompt) > 80 else ''}",
                ),
                unsafe_allow_html=True,
            )

        # Update score overlay + breakdown on right
        with right:
            score = result.score
            improving = ""
            prev_score = st.session_state.get("best_score", 0.0)
            if i > 0:
                improving = " improving ↑" if score >= prev_score else " ↓"
            score_container.markdown(
                f'<div class="score-display">'
                f'<div><div class="score-big {_score_class(score)}">{score:.1f}</div>'
                f'<div class="score-label">taste match score</div></div>'
                f'<div style="text-align:right;font-family:monospace;font-size:11px;color:#555;">'
                f"iteration {i + 1} of max {MAX_ITERATIONS}"
                f'<div style="color:#c8a96e;font-size:10px;margin-top:2px;">{improving}</div>'
                f"</div></div>",
                unsafe_allow_html=True,
            )
            # Breakdown (show when score >= 5)
            if score >= 5.0:
                rows = []
                for k in CRITIQUE_DIMENSIONS:
                    v = result.breakdown.get(k, 0.0)
                    sym = _score_symbol(v)
                    css = "breakdown-" + ("high" if v >= 8 else "mid" if v >= 5 else "low")
                    label = k.replace("_", " ")
                    rows.append(
                        f'<div class="breakdown-row">'
                        f'<span class="breakdown-key">{label}</span>'
                        f'<span class="{css}">{sym} {v:.0f}/10</span>'
                        f"</div>"
                    )
                breakdown_container.markdown(
                    '<div style="background:#141414;border:1px solid #1e1e1e;padding:14px;margin-top:8px;">'
                    '<div class="section-label" style="margin-bottom:8px;">last critique breakdown</div>'
                    + "".join(rows) + "</div>",
                    unsafe_allow_html=True,
                )

            status_container.empty()

        if result.score >= CONVERGENCE_THRESHOLD:
            st.session_state["final_image"] = image_bytes
            st.session_state["final_result"] = result
            st.session_state["phase"] = Phase.DONE
            st.rerun()
            return

        if result.score >= HUMAN_GATE_LOW:
            st.session_state["pending_gate"] = {
                "image": image_bytes,
                "iteration": i,
                "result": result,
            }
            st.stop()
            return

        # Update prompt display
        prompt = result.revised_prompt
        with left:
            prompt_container.caption(f"`{prompt[:120]}{'...' if len(prompt) > 120 else ''}`")

    # Exhausted — use best seen
    best = st.session_state.get("best_image")
    best_score = st.session_state.get("best_score", 0.0)
    with right:
        status_container.warning(
            f"Best effort after {MAX_ITERATIONS} iterations (score: {best_score:.1f}/10)"
        )
    st.session_state["final_image"] = best
    st.session_state["phase"] = Phase.DONE
    st.rerun()


def render_done() -> None:
    render_step_trail()
    final = st.session_state.get("final_image")
    final_result = st.session_state.get("final_result")
    profile: TasteProfile = st.session_state.get("taste_profile")

    left, right = st.columns([1, 1])

    with right:
        if final:
            st.image(final, use_container_width=True)

        if final_result:
            score = final_result.score
            st.markdown(
                f'<div class="score-display">'
                f'<div><div class="score-big {_score_class(score)}">{score:.1f}</div>'
                f'<div class="score-label">taste match score</div></div>'
                f'<div style="font-family:monospace;font-size:11px;color:#4a8;">converged ✓</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            render_breakdown(final_result)

        st.markdown("<br>", unsafe_allow_html=True)
        bcol1, bcol2 = st.columns(2)

        if final:
            bcol1.download_button(
                "approve & download image",
                data=final,
                file_name="air_generated.png",
                mime="image/png",
                use_container_width=True,
                type="primary",
            )

        if profile:
            bcol2.download_button(
                "export taste.md",
                data=_taste_md(profile),
                file_name="taste.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with left:
        if profile:
            st.markdown('<div class="section-label">what i understood about your taste</div>', unsafe_allow_html=True)
            for p in profile.principles:
                st.markdown(f'<div class="principle">{p}</div>', unsafe_allow_html=True)
            avoid = "; ".join(profile.avoid_list)
            st.markdown(f'<div class="principle negative">AVOID: {avoid}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption(f"Palette: {profile.palette_tendency}")
            st.caption(f"Composition: {profile.compositional_style}")
            st.caption(f"Mood: {profile.emotional_register}")
            st.caption(f"Texture: {profile.texture_feel}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺ start over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ── Main dispatch ─────────────────────────────────────────────────────────────
phase = st.session_state["phase"]

if phase == Phase.UPLOAD:
    render_upload()
elif phase == Phase.CONFIRM:
    render_confirm()
elif phase == Phase.GENERATING:
    render_generating()
elif phase == Phase.DONE:
    render_done()
