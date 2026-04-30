# AIR Generator — Implementation Spec

**What it does:** Takes 3–5 reference images representing a user's aesthetic taste, extracts the underlying principles via vision-LLM, generates art matching those principles, and critiques each generation against the taste profile in a loop until convergence.

**Stack:** Python 3.13 · Streamlit 1.57+ · openai 2.x · instructor (JSON mode) · Pillow

---

## Critical: How Streamlit Executes

Before writing a single line of UI code, internalize this: **Streamlit re-runs the entire `main.py` script from top to bottom on every user interaction** — every button click, every widget change, every file upload.

Consequences:

**1. Never store class instances in session_state if the class is defined in the script.**
Every rerun creates a new class object. Identity comparison (`stored_value == CurrentClass.MEMBER`) will always be `False` because they belong to different class objects. Use plain strings or integers for any value that lives in session_state across reruns.

```python
# WRONG — Phase class is redefined on every rerun
class Phase(Enum):
    UPLOAD = auto()

st.session_state["phase"] = Phase.UPLOAD
# On next rerun: st.session_state["phase"] == Phase.UPLOAD → False

# CORRECT — strings survive reruns unchanged
PHASE_UPLOAD = "upload"
st.session_state["phase"] = PHASE_UPLOAD
# On next rerun: st.session_state["phase"] == PHASE_UPLOAD → True
```

**2. File objects from `st.file_uploader` have a read pointer.** Reading a file object (for display OR for processing) advances the pointer. Any subsequent read returns empty bytes. Read all bytes exactly once, immediately, before any other use:

```python
# CORRECT — read once, reuse the bytes everywhere
files = st.file_uploader(...)
if files:
    file_data = [(f.name, f.read()) for f in files]   # read once here
    for col, (name, data) in zip(cols, file_data):
        col.image(data)                                # use bytes, not f
    # later: write data to tmp file, store in session_state, etc.
```

**3. Widget-bound session_state keys are owned by Streamlit.** If you create a widget with `key="foo"`, Streamlit manages `session_state["foo"]` automatically. You cannot call `st.session_state["foo"] = value` in the same run — this raises `StreamlitAPIException`. Read the value from session_state; never set it manually.

```python
st.text_input("Intent", key="user_intent")     # Streamlit writes session_state["user_intent"]
# ...
if st.button("Go"):
    intent = st.session_state["user_intent"]   # read it — do NOT set it
    st.session_state["phase"] = PHASE_NEXT     # set a different key — fine
```

**4. Local variables reset on every rerun.** Any variable that must survive a `st.stop()` / `st.rerun()` / human gate interaction must live in session_state:

```python
# WRONG — resets to None on every rerun
best_image = None

# CORRECT — survives human gate reruns
if "best_image" not in st.session_state:
    st.session_state["best_image"] = None
```

**5. Pre-allocate `st.empty()` containers before any loop.** Creating containers inside a loop adds new elements on every iteration instead of updating in place.

```python
# CORRECT
iter_slots = [st.empty() for _ in range(MAX_ITERATIONS)]
for i in range(MAX_ITERATIONS):
    iter_slots[i].markdown(...)   # updates in place
```

---

## Phase State Machine

Use string constants for phase. Initialize once in session_state. Dispatch at the bottom of the script.

```python
PHASE_UPLOAD     = "upload"
PHASE_CONFIRM    = "confirm"
PHASE_GENERATING = "generating"
PHASE_DONE       = "done"

if "phase" not in st.session_state:
    st.session_state["phase"] = PHASE_UPLOAD

# Human gate check runs BEFORE phase dispatch (see below)

phase = st.session_state["phase"]
if phase == PHASE_UPLOAD:     render_upload()
elif phase == PHASE_CONFIRM:  render_confirm()
elif phase == PHASE_GENERATING: render_generating()
elif phase == PHASE_DONE:     render_done()
```

Flow:
```
UPLOAD → [Extract taste button] → CONFIRM → [Confirm & generate button] → GENERATING
                                                                               ↓
                                                              score ≥ 8.0 → DONE
                                                              6.0 ≤ score < 8.0 → human gate
                                                              ↑ [No — keep iterating]
```

---

## API Setup

### Shared client (`client.py`)

```python
import os, openai, instructor
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL = "google/gemini-3.1-flash-image-preview"

_raw = openai.OpenAI(
    base_url=os.environ["BASE_URL"].strip().strip('"'),
    api_key=os.environ["API_KEY"].strip().strip('"'),
)

# For structured output: ALWAYS specify mode=instructor.Mode.JSON
# OpenAI-compatible proxy APIs typically do not support Mode.TOOLS (function calling)
instructor_client = instructor.from_openai(_raw, mode=instructor.Mode.JSON)
raw_client = _raw
```

### Image generation response format

Before writing `generator/client.py`, run this probe and read the output:

```python
resp = raw_client.chat.completions.create(
    model=BASE_MODEL,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Generate a test image of a red circle"}
    ]}],
)
import json; print(json.dumps(resp.model_dump(), indent=2))
```

For `google/gemini-3.1-flash-image-preview` via tokenrouter, the image arrives in a non-standard location — **not** in `content` (which is `None`) but in `model_extra`:

```python
images = resp.choices[0].message.model_extra.get("images", [])
data_url = images[0]["image_url"]["url"]          # "data:image/png;base64,..."
image_bytes = base64.b64decode(data_url.split(",", 1)[1])
```

Always write the parser defensively — check `isinstance`, handle both URL and base64, raise a descriptive `ValueError` with `resp.model_dump()` if nothing matches.

---

## Data Models (`taste/profile.py`)

```python
from pydantic import BaseModel, field_validator

CRITIQUE_DIMENSIONS = [
    "palette_tendency",
    "compositional_style",
    "emotional_register",
    "texture_feel",
    "avoid_list",
]

class TasteProfile(BaseModel):
    principles: list[str]          # overarching aesthetic principles
    palette_tendency: str
    compositional_style: str
    emotional_register: str
    texture_feel: str
    avoid_list: list[str]

    @field_validator("principles")
    @classmethod
    def principles_not_empty(cls, v):
        if not v: raise ValueError("principles cannot be empty")
        return v

class CritiqueResult(BaseModel):
    score: float                   # 0–10
    breakdown: dict[str, float]    # keys must be CRITIQUE_DIMENSIONS
    reasoning: str                 # one sentence, biggest gap
    revised_prompt: str            # improved prompt for next iteration

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v):
        if not 0 <= v <= 10: raise ValueError("score must be 0-10")
        return v
```

---

## Initial Prompt (`loop.py`)

Include **all** TasteProfile fields, including `principles`. Separate list items with `"; "` so the model can parse them as distinct constraints:

```python
def initial_prompt(user_intent: str, profile: TasteProfile) -> str:
    return (
        f"{user_intent}. "
        f"Taste principles: {'; '.join(profile.principles)}. "
        f"Palette: {profile.palette_tendency}. "
        f"Composition: {profile.compositional_style}. "
        f"Mood: {profile.emotional_register}. "
        f"Texture: {profile.texture_feel}. "
        f"Avoid: {', '.join(profile.avoid_list)}."
    )
```

---

## Critique Loop (`loop.py`)

**No Streamlit imports.** The loop receives callbacks via dependency injection so it can be tested headlessly.

```python
CONVERGENCE_THRESHOLD = 8.0
HUMAN_GATE_LOW        = 6.0   # gate fires when HUMAN_GATE_LOW ≤ score < CONVERGENCE_THRESHOLD
MAX_ITERATIONS        = 5

def run(prompt, taste_profile, generator, critic,
        on_iteration_complete, on_human_gate,
        max_iterations=MAX_ITERATIONS,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        human_gate_low=HUMAN_GATE_LOW) -> LoopResult:

    if human_gate_low >= convergence_threshold:
        raise ValueError("human_gate_low must be < convergence_threshold")

    best_image, best_score = b"", 0.0

    for i in range(max_iterations):
        image = generator.generate(prompt)             # raises on failure

        try:
            result = critic.evaluate(image, taste_profile)
        except Exception as e:
            # Critic failed — report the failure, continue with unchanged prompt
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
            if on_human_gate(image, result):           # True = approved
                return LoopResult(image, result.score, i + 1,
                                  converged=False, human_approved=True)

        prompt = result.revised_prompt

    return LoopResult(best_image, best_score, max_iterations, converged=False)
```

---

## Critique Format (`loop.py`)

Iterate `CRITIQUE_DIMENSIONS` in declared order (not `result.breakdown.items()` which is unordered):

```python
def format_critique(i: int, result: CritiqueResult) -> str:
    dims = " | ".join(
        f"{k}: {result.breakdown.get(k, 0):.1f}" for k in CRITIQUE_DIMENSIONS
    )
    return (
        f"**Iteration {i + 1}** — score {result.score:.1f}/10\n"
        f"{dims}\n_{result.reasoning}_\n→ {result.revised_prompt}"
    )
```

---

## Human Gate Pattern (`main.py`)

Check for `pending_gate` **before** the phase dispatch, at the top of `main.py`. Use `st.stop()` to pause execution:

```python
# TOP OF main.py — before phase dispatch
if "pending_gate" in st.session_state:
    gate = st.session_state["pending_gate"]
    # render gate UI...
    col1, col2, col3 = st.columns(3)
    if col1.button("yes — approve"):
        st.session_state["final_image"] = gate["image"]
        del st.session_state["pending_gate"]
        st.session_state["phase"] = PHASE_DONE
        st.rerun()
    if col2.button("almost — keep iterating"):
        st.session_state["resume_prompt"] = gate["result"].revised_prompt
        del st.session_state["pending_gate"]
        st.rerun()
    if col3.button("no — restart"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
    st.stop()   # don't render the rest of the app while gate is open

# THEN: phase dispatch
phase = st.session_state["phase"]
...
```

Inside `render_generating()`, trigger the gate and pause:

```python
st.session_state["pending_gate"] = {"image": image_bytes, "iteration": i, "result": result}
st.stop()
return
```

On the next rerun, the gate handler at the top picks it up. If the user rejects, `resume_prompt` is stored. The generating render should consume it:

```python
prompt = st.session_state.pop("resume_prompt", None) or initial_prompt(user_intent, profile)
```

---

## Extraction Prompt (`taste/extractor.py`)

```
You are analyzing a collection of {N} images that represent someone's aesthetic taste.
For each dimension below, write ONE precise, opinionated sentence describing what these
images share — not what they depict, but the underlying aesthetic choice:

1. Palette tendency (color relationships, saturation, temperature)
2. Compositional style (framing, balance, negative space)
3. Emotional register (the feeling evoked, not the subject matter)
4. Texture / medium feel (analog vs digital, grain, materiality)
5. Avoid-list: what does this collection deliberately reject? (3–5 items)

Then write 3–5 overarching principles that unify the above.
Be specific enough that an image generation model could use this as a prompt constraint.
Describe the pattern across all images — never describe a single image.
```

Resize all images to max 1024px before sending to extractor. Resize to max 512px before sending to critic.

---

## Build Order

1. `taste/profile.py` + `tests/test_helpers.py` — models, validators, helper functions. **Write tests first.**
2. `client.py` — shared API client. **Smoke-test the connection and run the image response probe before writing anything else.**
3. `taste/extractor.py` — test on real reference images; if the extracted principles feel generic, fix the prompt before proceeding.
4. `taste/critic.py`
5. `generator/client.py` — implement based on probe findings from step 2.
6. `loop.py` — run headlessly (no Streamlit) to verify convergence logic.
7. `main.py` — Streamlit UI last.

---

## Avoid-list Widget Key

The field `avoid_list` on `TasteProfile` conflicts with Streamlit's session_state if used as a widget key. Use a distinct key for the widget:

```python
st.text_area("Avoid list", value=", ".join(profile.avoid_list), key="avoid_list_text")
# reconstruct:
avoid_list = [s.strip() for s in st.session_state["avoid_list_text"].split(",") if s.strip()]
```

---

## Testing

```bash
python3.13 -m pytest tests/ -v    # 6 pure-function tests, no API calls needed
streamlit run main.py             # manual smoke test
```

Pure function tests cover: `initial_prompt` includes all principles, `format_critique` rounds score and respects CRITIQUE_DIMENSIONS order, `TasteProfile` rejects empty principles, `CritiqueResult` rejects out-of-range scores.
