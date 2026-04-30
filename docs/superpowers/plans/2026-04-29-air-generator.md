# AIR Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a taste-aware AI art generator that extracts aesthetic principles from reference images, generates art matching those principles, and iterates with a vision-LLM critic until the result is "spookily accurate."

**Architecture:** Streamlit app with 4 phases (UPLOAD → CONFIRM → GENERATING → DONE). Pure Python loop with dependency injection (no Streamlit in loop.py). Single model `google/gemini-3.1-flash-image-preview` via tokenrouter for extraction, generation, and critique.

**Tech Stack:** Python 3.12, Streamlit, openai 2.x, instructor, Pillow, pytest, python-dotenv

---

## File Map

```
agent-hack-day/
├── .env                        — BASE_URL + API_KEY (exists)
├── requirements.txt            — CREATE: pinned deps
├── client.py                   — CREATE: shared openai + instructor singleton
├── taste/
│   ├── __init__.py             — CREATE: empty
│   ├── profile.py              — CREATE: TasteProfile, CritiqueResult, CRITIQUE_DIMENSIONS, validators
│   ├── extractor.py            — CREATE: extract(image_paths) → TasteProfile
│   └── critic.py               — CREATE: evaluate(image_bytes, profile) → CritiqueResult
├── generator/
│   ├── __init__.py             — CREATE: empty
│   └── client.py               — CREATE: generate(prompt) → bytes (Gemini image output)
├── loop.py                     — CREATE: run(..., callbacks) → LoopResult, initial_prompt(), format_critique()
├── main.py                     — CREATE: Streamlit app, Phase state machine
└── tests/
    ├── __init__.py             — CREATE: empty
    └── test_helpers.py         — CREATE: 6 pure function tests (TDD first)
```

---

## Task 0: Install Dependencies + Project Init

**Files:**
- Create: `requirements.txt`
- Create: `taste/__init__.py`
- Create: `generator/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
streamlit>=1.45.0
openai>=2.32.0
instructor>=1.9.0
pillow>=11.0.0
python-dotenv>=1.2.2
pytest>=9.0.0
```

- [ ] **Step 2: Install dependencies**

```bash
cd /home/dev/.claude/repos/agent-hack-day
pip3 install streamlit instructor pillow
```

Expected: No errors. `pip3 show streamlit instructor pillow` shows versions.

- [ ] **Step 3: Create package init files**

```bash
touch taste/__init__.py generator/__init__.py tests/__init__.py
```

- [ ] **Step 4: Verify .env has correct keys**

Check `.env` contains:
```
BASE_URL = "https://api.tokenrouter.com/v1"
API_KEY = "sk-..."
```

Note: The .env uses spaces around `=`. Use `python-dotenv` with `load_dotenv()` — it handles this correctly.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt taste/__init__.py generator/__init__.py tests/__init__.py .env
git commit -m "chore: project structure and dependencies"
```

---

## Task 1: Data Models (TDD First)

**Files:**
- Create: `tests/test_helpers.py`
- Create: `taste/profile.py`

- [ ] **Step 1: Write all 6 failing tests**

Create `tests/test_helpers.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ALL to fail (imports don't exist yet)**

```bash
cd /home/dev/.claude/repos/agent-hack-day
pytest tests/test_helpers.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'taste'` or similar.

- [ ] **Step 3: Create taste/profile.py**

```python
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
```

- [ ] **Step 4: Create a stub loop.py with just the two helper functions (tests need them)**

```python
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
            # Generation failed — skip iteration, use best seen
            if best_image is not None:
                return LoopResult(best_image, best_score, i, False)
            raise

        try:
            result = critic.evaluate(image, taste_profile)
        except Exception:
            # Critic failed — retry once with same prompt
            try:
                result = critic.evaluate(image, taste_profile)
            except Exception:
                prompt = prompt  # unchanged
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
```

- [ ] **Step 5: Run tests — all 6 should pass**

```bash
pytest tests/test_helpers.py -v
```

Expected output:
```
PASSED tests/test_helpers.py::test_initial_prompt_includes_principles
PASSED tests/test_helpers.py::test_initial_prompt_formats_avoid_list
PASSED tests/test_helpers.py::test_format_critique_rounds_score
PASSED tests/test_helpers.py::test_format_critique_iterates_critique_dimensions_order
PASSED tests/test_helpers.py::test_taste_profile_rejects_empty_principles
PASSED tests/test_helpers.py::test_critique_result_rejects_out_of_range_score
6 passed
```

- [ ] **Step 6: Commit**

```bash
git add taste/profile.py loop.py tests/test_helpers.py tests/__init__.py
git commit -m "feat: data models, loop helpers, and passing tests"
```

---

## Task 2: Shared Client

**Files:**
- Create: `client.py`

- [ ] **Step 1: Create client.py**

```python
from __future__ import annotations
import os
import openai
import instructor
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL = "google/gemini-3.1-flash-image-preview"

_openai_client = openai.OpenAI(
    base_url=os.environ["BASE_URL"].strip().strip('"'),
    api_key=os.environ["API_KEY"].strip().strip('"'),
)

# Instructor-wrapped for structured output (extractor, critic)
instructor_client: instructor.Instructor = instructor.from_openai(_openai_client)

# Raw client for image generation and probing
raw_client: openai.OpenAI = _openai_client
```

Note: `.strip().strip('"')` handles the quoted values in the .env file (`BASE_URL = "https://..."`).

- [ ] **Step 2: Smoke-test the client**

```bash
cd /home/dev/.claude/repos/agent-hack-day
python3 -c "
from client import raw_client, BASE_MODEL
resp = raw_client.chat.completions.create(
    model=BASE_MODEL,
    messages=[{'role': 'user', 'content': 'Reply with just: OK'}],
    max_tokens=5,
)
print(resp.choices[0].message.content)
"
```

Expected: `OK` (or similar short response). If you get an auth error, check the API_KEY in `.env`.

- [ ] **Step 3: Probe Gemini image output format**

This is critical — the exact response structure for image generation is non-standard. Run this probe and note the structure:

```bash
python3 -c "
from client import raw_client, BASE_MODEL
import json
resp = raw_client.chat.completions.create(
    model=BASE_MODEL,
    messages=[{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Generate a simple image of a red circle on white background'}
    ]}],
)
print(json.dumps(resp.model_dump(), indent=2))
" 2>&1 | head -60
```

Look for where the image appears in the JSON. Common patterns:
- `choices[0].message.content` is a list with an `image_url` block: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`
- Or `choices[0].message.content` is a string containing a URL

The generator/client.py in Task 4 handles both.

- [ ] **Step 4: Commit**

```bash
git add client.py
git commit -m "feat: shared tokenrouter client (openai + instructor)"
```

---

## Task 3: Taste Extractor

**Files:**
- Create: `taste/extractor.py`

- [ ] **Step 1: Create taste/extractor.py**

```python
from __future__ import annotations
import base64
import io
from pathlib import Path

from PIL import Image as PILImage

from client import instructor_client, BASE_MODEL
from taste.profile import TasteProfile

EXTRACTION_PROMPT = """You are analyzing a collection of images that represent someone's aesthetic taste.
For each of the following dimensions, write one precise, opinionated sentence that captures
what these images share — not what they depict, but what underlies the aesthetic choices:

1. Palette tendency (color relationships, saturation, temperature)
2. Compositional style (framing, balance, negative space use)
3. Emotional register (the feeling evoked, not the subject matter)
4. Texture / medium feel (analog vs digital, grain, smoothness, materiality)
5. Avoid-list: what does this collection deliberately reject? (3-5 items)

Then write 3-5 overarching principles that unify all the above.
Be specific enough that an image generation model could use this as a prompt constraint.
Avoid describing individual images — only describe the pattern across all of them."""


def _resize_to_bytes(path: str | Path, max_size: int = 1024) -> bytes:
    img = PILImage.open(path)
    img.thumbnail((max_size, max_size), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _to_base64_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:image/jpeg;base64,{b64}"


def extract(image_paths: list[str | Path]) -> TasteProfile:
    """Extract a TasteProfile from 3-5 reference images."""
    if len(image_paths) < 3:
        raise ValueError(f"Need at least 3 images, got {len(image_paths)}")

    content: list[dict] = [{"type": "text", "text": EXTRACTION_PROMPT}]
    for path in image_paths:
        img_bytes = _resize_to_bytes(path, max_size=1024)
        content.append({
            "type": "image_url",
            "image_url": {"url": _to_base64_url(img_bytes)},
        })

    return instructor_client.chat.completions.create(
        model=BASE_MODEL,
        response_model=TasteProfile,
        messages=[{"role": "user", "content": content}],
        max_retries=2,
    )
```

- [ ] **Step 2: Test extraction on the Marc Bolan images**

```bash
cd /home/dev/.claude/repos/agent-hack-day
python3 -c "
from taste.extractor import extract
from pathlib import Path

imgs = sorted(Path('img_set_1').glob('*'))
print(f'Extracting from {len(imgs)} images...')
profile = extract(imgs)
print()
print('=== TASTE PROFILE ===')
print('Principles:')
for p in profile.principles:
    print(f'  - {p}')
print(f'Palette: {profile.palette_tendency}')
print(f'Composition: {profile.compositional_style}')
print(f'Mood: {profile.emotional_register}')
print(f'Texture: {profile.texture_feel}')
print(f'Avoid: {profile.avoid_list}')
"
```

Expected: A TasteProfile that captures Marc Bolan's aesthetic — glam rock, bold color, theatrical, etc. If the output feels generic (e.g., "varied palette"), the extraction prompt needs tuning. Read it out loud. Does it feel right?

- [ ] **Step 3: Commit**

```bash
git add taste/extractor.py
git commit -m "feat: taste extractor (Gemini vision → TasteProfile)"
```

---

## Task 4: Image Generator

**Files:**
- Create: `generator/client.py`

Note: Complete Task 2 Step 3 (probe) first — the parser logic below depends on what you saw.

- [ ] **Step 1: Create generator/client.py**

```python
from __future__ import annotations
import base64
import re
import io
import requests

from PIL import Image as PILImage
from client import raw_client, BASE_MODEL


def _extract_image_from_response(resp) -> bytes:
    """Parse Gemini image output from chat completions response.

    Handles three formats:
    1. content is list with image_url block (base64 data URL)
    2. content is list with image_url block (https URL)
    3. content is plain string (fallback — shouldn't happen)
    """
    content = resp.choices[0].message.content

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                url = block["image_url"]["url"]
                if url.startswith("data:"):
                    # Base64 data URL: data:image/png;base64,<data>
                    b64_data = url.split(",", 1)[1]
                    return base64.b64decode(b64_data)
                else:
                    # Regular URL — fetch it
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    return r.content
            # Also handle openai SDK objects (not plain dicts)
            if hasattr(block, "type") and block.type == "image_url":
                url = block.image_url.url
                if url.startswith("data:"):
                    b64_data = url.split(",", 1)[1]
                    return base64.b64decode(b64_data)
                else:
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    return r.content

    # Fallback: look for a URL in the text content
    text = content if isinstance(content, str) else str(content)
    url_match = re.search(r"https?://\S+\.(?:png|jpg|jpeg|webp)", text)
    if url_match:
        r = requests.get(url_match.group(), timeout=30)
        r.raise_for_status()
        return r.content

    raise ValueError(f"Could not extract image from response: {resp.model_dump()}")


def generate(prompt: str) -> bytes:
    """Generate an image from a taste-encoded prompt. Returns raw image bytes."""
    resp = raw_client.chat.completions.create(
        model=BASE_MODEL,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": f"Generate an image: {prompt}"}],
        }],
    )
    return _extract_image_from_response(resp)
```

- [ ] **Step 2: Test image generation**

```bash
cd /home/dev/.claude/repos/agent-hack-day
python3 -c "
from generator.client import generate

print('Generating test image...')
img_bytes = generate('a moody street at dusk, analog film grain, warm amber tones, asymmetric composition')
print(f'Got {len(img_bytes)} bytes')

# Save to disk to inspect
with open('/tmp/test_gen.jpg', 'wb') as f:
    f.write(img_bytes)
print('Saved to /tmp/test_gen.jpg')
"
```

Expected: `Got XXXXX bytes` (anything > 1000). If you get `ValueError: Could not extract image`, print the full response with `resp.model_dump()` and adjust `_extract_image_from_response` to match the actual structure.

- [ ] **Step 3: Commit**

```bash
git add generator/client.py
git commit -m "feat: image generator (Gemini chat completions → bytes)"
```

---

## Task 5: Critic

**Files:**
- Create: `taste/critic.py`

- [ ] **Step 1: Create taste/critic.py**

```python
from __future__ import annotations
import base64
import io

from PIL import Image as PILImage

from client import instructor_client, BASE_MODEL
from taste.profile import TasteProfile, CritiqueResult, CRITIQUE_DIMENSIONS


def _resize_bytes(image_bytes: bytes, max_size: int = 512) -> bytes:
    img = PILImage.open(io.BytesIO(image_bytes))
    img.thumbnail((max_size, max_size), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _to_base64_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:image/jpeg;base64,{b64}"


def _build_critique_prompt(profile: TasteProfile) -> str:
    dims = "\n".join(f"- {d}: {getattr(profile, d)}" for d in CRITIQUE_DIMENSIONS[:-1])
    return f"""You are a taste critic. Score this generated image against a specific aesthetic taste profile.

TASTE PROFILE:
Principles:
{chr(10).join(f'- {p}' for p in profile.principles)}

Dimensions:
{dims}
- avoid_list: avoid {', '.join(profile.avoid_list)}

Score each dimension 0-10 (how well the image matches that dimension of the taste).
Give an overall score 0-10.
In one sentence, name the biggest gap.
Write an improved generation prompt that addresses the biggest gap while keeping everything else.

Return a JSON object with keys: score, breakdown (dict with keys: {CRITIQUE_DIMENSIONS}), reasoning, revised_prompt."""


def evaluate(image_bytes: bytes, profile: TasteProfile) -> CritiqueResult:
    """Score a generated image against a TasteProfile. Returns CritiqueResult."""
    small = _resize_bytes(image_bytes, max_size=512)
    b64_url = _to_base64_url(small)

    prompt = _build_critique_prompt(profile)

    return instructor_client.chat.completions.create(
        model=BASE_MODEL,
        response_model=CritiqueResult,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": b64_url}},
            ],
        }],
        max_retries=2,
    )
```

- [ ] **Step 2: Test the critic end-to-end with a generated image**

```bash
cd /home/dev/.claude/repos/agent-hack-day
python3 -c "
from pathlib import Path
from taste.extractor import extract
from generator.client import generate
from taste.critic import evaluate
from loop import initial_prompt

imgs = sorted(Path('img_set_1').glob('*'))
print('Extracting taste...')
profile = extract(imgs)
print(f'Got profile. Principles: {profile.principles[:1]}')

prompt = initial_prompt('a portrait of a musician on stage', profile)
print(f'Generating with prompt: {prompt[:80]}...')
img_bytes = generate(prompt)
print(f'Got {len(img_bytes)} bytes. Critiquing...')

result = evaluate(img_bytes, profile)
print(f'Score: {result.score}/10')
print(f'Breakdown: {result.breakdown}')
print(f'Reasoning: {result.reasoning}')
print(f'Revised: {result.revised_prompt[:80]}')
"
```

Expected: A score between 0-10, breakdown dict with all CRITIQUE_DIMENSIONS keys, a reasoning sentence, and a revised prompt.

- [ ] **Step 3: Commit**

```bash
git add taste/critic.py
git commit -m "feat: taste critic (Gemini vision → CritiqueResult)"
```

---

## Task 6: Loop Orchestration (update loop.py)

The stub loop.py from Task 1 already has the full `run()` implementation. This task verifies it works end-to-end without Streamlit.

**Files:**
- Modify: `loop.py` (add import for generator and test)

- [ ] **Step 1: Verify loop.py imports are correct (no Streamlit)**

```bash
cd /home/dev/.claude/repos/agent-hack-day
python3 -c "import loop; print('loop imports OK')"
```

Expected: `loop imports OK` with no errors.

- [ ] **Step 2: Run a headless loop test (no Streamlit)**

```bash
python3 -c "
from pathlib import Path
from taste.extractor import extract
from taste import critic as critic_module
from generator import client as generator_module
from loop import run, initial_prompt
from taste.profile import TasteProfile

imgs = sorted(Path('img_set_1').glob('*'))
print('Extracting taste...')
profile = extract(imgs)

prompt = initial_prompt('a street at dusk, empty', profile)

results = []
def on_complete(i, image, critique):
    results.append((i, critique.score))
    print(f'  Iteration {i+1}: score={critique.score:.1f}')

def on_gate(image, critique):
    print(f'  Human gate at score {critique.score:.1f} — auto-approving for test')
    return True  # auto-approve

print('Running loop (max 2 iterations for speed)...')
result = run(
    prompt=prompt,
    taste_profile=profile,
    generator=generator_module,
    critic=critic_module,
    on_iteration_complete=on_complete,
    on_human_gate=on_gate,
    max_iterations=2,
)
print(f'Done. converged={result.converged}, score={result.final_score:.1f}, iterations={result.iterations}')
print(f'Image: {len(result.final_image)} bytes')
"
```

Expected: 1-2 iterations run, score printed each time, final result with image bytes > 0.

- [ ] **Step 3: Run all tests to make sure nothing broke**

```bash
pytest tests/ -v
```

Expected: All 6 tests pass.

- [ ] **Step 4: Commit**

```bash
git add loop.py
git commit -m "feat: loop orchestration verified headless"
```

---

## Task 7: Streamlit UI (main.py)

**Files:**
- Create: `main.py`

The UI follows the wireframe in `wireframe.html`: two-panel layout, left panel shows the loop, right panel shows the image. Dark theme. Phase state machine: UPLOAD → CONFIRM → GENERATING → DONE.

- [ ] **Step 1: Create main.py**

```python
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
    LoopResult,
    format_critique,
    initial_prompt,
    run,
)
from taste import critic as critic_module
from taste import extractor
from generator import client as generator_module
from taste.profile import TasteProfile


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIR Generator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
body { background: #0d0d0d; color: #e0e0e0; }
.stApp { background: #0d0d0d; }
</style>
""", unsafe_allow_html=True)


# ── Phase state machine ───────────────────────────────────────────────────────
class Phase(Enum):
    UPLOAD = auto()
    CONFIRM = auto()
    GENERATING = auto()
    DONE = auto()


if "phase" not in st.session_state:
    st.session_state["phase"] = Phase.UPLOAD


# ── Human gate handler (runs before any phase rendering) ──────────────────────
if "pending_gate" in st.session_state:
    gate = st.session_state["pending_gate"]
    st.title("AIR — does this feel right?")
    st.image(gate["image"], use_container_width=True)
    st.info(
        f"Iteration {gate['iteration'] + 1}: score {gate['result'].score:.1f}/10 "
        f"(uncertain range). Does this feel right?"
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


# ── Render functions ──────────────────────────────────────────────────────────

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
                # Save uploaded files to temp paths
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
            key="avoid_list",
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

        # Reconstruct TasteProfile from widget values
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
            "avoid_list": [s.strip() for s in st.session_state["avoid_list"].split(",") if s.strip()],
        })
        st.session_state["taste_profile"] = updated
        st.session_state["user_intent"] = user_intent.strip()
        st.session_state["phase"] = Phase.GENERATING
        # Initialize best tracking
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
                if st.session_state.get("best_image"):
                    st.session_state["final_image"] = st.session_state["best_image"]
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
                    prompt = prompt
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


# ── Main dispatch ──────────────────────────────────────────────────────────────
phase = st.session_state["phase"]

if phase == Phase.UPLOAD:
    render_upload()
elif phase == Phase.CONFIRM:
    render_confirm()
elif phase == Phase.GENERATING:
    render_generating()
elif phase == Phase.DONE:
    render_done()
```

- [ ] **Step 2: Run the app**

```bash
cd /home/dev/.claude/repos/agent-hack-day
streamlit run main.py --server.port 8501
```

Open http://localhost:8501 in a browser. Walk through the flow manually:
1. Upload the 5 Marc Bolan images from `img_set_1/`
2. Click "Extract taste →" — wait for extraction (~20s)
3. Review the extracted principles — do they feel like Marc Bolan's aesthetic?
4. Enter "a portrait of a glam rock musician, on stage" as the subject
5. Click "Confirm taste & generate →"
6. Watch iterations run in the left panel, image update on the right
7. Verify the loop runs without crashing

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: Streamlit UI with phase state machine (UPLOAD→CONFIRM→GENERATING→DONE)"
```

---

## Task 8: End-to-End Verification

Run all tests and do a final manual smoke-test.

- [ ] **Step 1: Run full test suite**

```bash
cd /home/dev/.claude/repos/agent-hack-day
pytest tests/ -v
```

Expected: 6/6 tests pass.

- [ ] **Step 2: Manual end-to-end smoke test**

With `streamlit run main.py` running:

1. **Happy path:** Upload 3 images, extract, confirm, generate "a street in Tokyo at dusk, empty", watch loop converge or reach best-effort.
2. **Edge case:** Upload only 2 images → verify warning "Upload at least 3 images" appears and no extraction triggers.
3. **Restart:** Click "↺ Start over" from DONE phase → verify app returns to UPLOAD with clean state.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: AIR generator complete — taste extraction, generation, critic loop, Streamlit UI"
```

---

## Verification Summary

| Check | Command | Expected |
|---|---|---|
| Unit tests | `pytest tests/ -v` | 6/6 pass |
| Client smoke | `python3 -c "from client import raw_client; print('ok')"` | `ok` |
| Extraction | `python3 -c "from taste.extractor import extract; ..."` | TasteProfile printed |
| Generation | `python3 -c "from generator.client import generate; ..."` | bytes > 1000 |
| Critic | `python3 -c "from taste.critic import evaluate; ..."` | score 0-10 |
| Full app | `streamlit run main.py` | Opens at localhost:8501 |
| End-to-end | Upload Marc Bolan images, run loop | Image generated |
