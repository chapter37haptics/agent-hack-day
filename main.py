from __future__ import annotations
import io
from enum import Enum, auto
from pathlib import Path

import streamlit as st
from PIL import Image as PILImage

from loop import (
    CONVERGENCE_THRESHOLD,
    HUMAN_GATE_LOW,
    MAX_ITERATIONS,
    format_critique,
    initial_prompt,
)
from taste import critic as critic_module
from taste import extractor
from generator import client as generator_module
from taste.profile import TasteProfile


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIR Generator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp { background: #0d0d0d; color: #e0e0e0; }
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


# ── Human gate (checked before any phase render) ──────────────────────────────
if "pending_gate" in st.session_state:
    gate = st.session_state["pending_gate"]
    st.title("AIR — does this feel right?")
    st.image(gate["image"], use_container_width=True)
    st.info(
        f"Iteration {gate['iteration'] + 1}: score {gate['result'].score:.1f}/10 "
        "(uncertain range). Does this feel right?"
    )
    col1, col2 = st.columns(2)
    if col1.button("✓ Yes — approve this", use_container_width=True):
        st.session_state["final_image"] = gate["image"]
        del st.session_state["pending_gate"]
        st.session_state["phase"] = Phase.DONE
        st.rerun()
    if col2.button("✗ No — keep iterating", use_container_width=True):
        st.session_state["resume_prompt"] = gate["result"].revised_prompt
        del st.session_state["pending_gate"]
        st.rerun()
    st.stop()


# ── Phase renderers ────────────────────────────────────────────────────────────

def render_upload() -> None:
    st.title("AIR — Taste-Aware Image Generator")
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
                    st.session_state["phase"] = Phase.CONFIRM
                    st.rerun()
                except Exception as e:
                    st.error(f"Extraction failed: {e}")


def render_confirm() -> None:
    profile: TasteProfile = st.session_state["taste_profile"]

    st.title("AIR — Does this sound like you?")
    st.markdown("Edit any principles that feel off, then confirm.")

    st.subheader("Overarching principles")
    for i, p in enumerate(profile.principles):
        st.text_area(f"Principle {i + 1}", value=p, key=f"principle_{i}", height=60)

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Palette tendency", value=profile.palette_tendency, key="palette_tendency")
        st.text_input("Compositional style", value=profile.compositional_style, key="compositional_style")
        st.text_input("Emotional register", value=profile.emotional_register, key="emotional_register")
    with col2:
        st.text_input("Texture feel", value=profile.texture_feel, key="texture_feel")
        st.text_area(
            "Avoid list (comma-separated)",
            value=", ".join(profile.avoid_list),
            key="avoid_list_text",
            height=100,
        )

    st.subheader("What do you want to generate?")
    user_intent = st.text_input(
        "Describe the subject",
        placeholder="e.g. a street in Tokyo at dusk, empty",
        key="user_intent",
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


def render_generating() -> None:
    profile: TasteProfile = st.session_state["taste_profile"]
    user_intent: str = st.session_state["user_intent"]

    st.title("AIR — Generating")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Taste profile")
        for p in profile.principles:
            st.markdown(f"- {p}")
        st.caption(f"Palette: {profile.palette_tendency}")
        st.caption(f"Mood: {profile.emotional_register}")
        st.caption(f"Avoid: {', '.join(profile.avoid_list)}")
        st.subheader("Critique log")
        iter_containers = [st.empty() for _ in range(MAX_ITERATIONS)]

    with right:
        image_container = st.empty()
        status_container = st.empty()

    prompt = st.session_state.pop("resume_prompt", None) or initial_prompt(user_intent, profile)

    for i in range(MAX_ITERATIONS):
        with st.spinner(f"Iteration {i + 1}/{MAX_ITERATIONS}: generating image..."):
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

        with st.spinner(f"Iteration {i + 1}/{MAX_ITERATIONS}: critiquing..."):
            try:
                result = critic_module.evaluate(image_bytes, profile)
            except Exception:
                try:
                    result = critic_module.evaluate(image_bytes, profile)
                except Exception as e:
                    with left:
                        iter_containers[i].warning(f"Iteration {i + 1}: critic failed ({e}), skipping")
                    continue

        if result.score > st.session_state.get("best_score", 0.0):
            st.session_state["best_image"] = image_bytes
            st.session_state["best_score"] = result.score

        with left:
            iter_containers[i].markdown(format_critique(i, result))

        if result.score >= CONVERGENCE_THRESHOLD:
            st.session_state["final_image"] = image_bytes
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

        prompt = result.revised_prompt

    # Exhausted without convergence
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
    st.title("AIR — Result")
    final = st.session_state.get("final_image")
    profile: TasteProfile = st.session_state.get("taste_profile")

    left, right = st.columns([1, 1])

    with right:
        if final:
            st.image(final, use_container_width=True)
        else:
            st.warning("No image was generated.")

    with left:
        if profile:
            st.subheader("What I understood about your taste")
            for p in profile.principles:
                st.markdown(f"- {p}")
            st.caption(f"Palette: {profile.palette_tendency}")
            st.caption(f"Composition: {profile.compositional_style}")
            st.caption(f"Mood: {profile.emotional_register}")
            st.caption(f"Texture: {profile.texture_feel}")
            st.caption(f"Avoid: {', '.join(profile.avoid_list)}")

        if st.button("↺ Start over", use_container_width=True):
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
