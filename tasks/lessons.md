# Lessons Learned

## Session: 2026-04-29

## Gemini image response structure (via tokenrouter)

When `google/gemini-3.1-flash-image-preview` generates an image:

- `choices[0].message.content` is `None` (not a string, not a list)
- `choices[0].finish_reason` is `"stop"`
- Image data lives in `choices[0].message.model_extra["images"]` — a list
- Each entry in `images` is a dict: `{"type": "image_url", "image_url": {"url": "<data_url>"}}`
- The `url` is a base64 data URL: `data:image/png;base64,<base64_data>` (mime type is `image/png`)
- There is NO https:// URL — it is always an inline base64 data URL

## Instructor mode (via tokenrouter)

- `instructor.from_openai(client)` defaults to `Mode.TOOLS` (function-calling)
- Tokenrouter returns 403 "Invalid authorization token" for tool_call requests
- Use `instructor.from_openai(client, mode=instructor.Mode.JSON)` instead
- JSON mode works: tokenrouter supports `response_format={"type": "json_object"}`

### Parsing pattern for generator/client.py
```python
images = resp.choices[0].message.model_extra.get("images", [])
if images:
    data_url = images[0]["image_url"]["url"]
    # data_url = "data:image/png;base64,iVBOR..."
    b64 = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(b64)
```

---

## Session: 2026-04-30 (retro learnings)

## Streamlit: never store Enum instances in session_state

Streamlit re-executes the entire script on every rerun, creating a **new Enum class each time**. The instance stored in `session_state` belongs to the old class, so `phase == Phase.UPLOAD` always returns `False` after the first rerun — nothing renders, page goes blank.

**Fix:** use plain string constants instead.

```python
# WRONG
class Phase(Enum):
    UPLOAD = auto()
st.session_state["phase"] = Phase.UPLOAD  # breaks on rerun

# CORRECT
PHASE_UPLOAD = "upload"
st.session_state["phase"] = PHASE_UPLOAD  # survives reruns
```

## Streamlit: read uploaded file bytes exactly once

`st.image(uploaded_file)` reads the file object's bytes, advancing its read pointer to the end. Any subsequent `f.read()` in the same rerun returns `b""`.

**Fix:** read all bytes upfront before any other use.

```python
file_data = [(f.name, f.read()) for f in files]  # read once here
col.image(data)                                    # use bytes, not f
tmp.write_bytes(data)                              # same bytes
```

## Streamlit: never set widget-bound session_state keys manually

If a widget is rendered with `key="foo"`, Streamlit owns `session_state["foo"]`. Calling `st.session_state["foo"] = value` in the same run raises `StreamlitAPIException`.

**Fix:** just read it — the widget already wrote it.

```python
st.text_input("Intent", key="user_intent")   # Streamlit manages this key
# ...
if st.button("Go"):
    intent = st.session_state["user_intent"]  # read — do NOT set
```

## Streamlit: local variables reset on every rerun

Any value that must survive a `st.stop()` / human gate / `st.rerun()` must live in `session_state`, not as a local variable.

```python
# WRONG — resets to None after human gate
best_image = None

# CORRECT
if "best_image" not in st.session_state:
    st.session_state["best_image"] = None
```

## Avoid-list widget key conflicts with model field names

Using `key="avoid_list"` for a widget conflicts with `TasteProfile.avoid_list` if the field name is later used as a session_state key elsewhere. Use a distinct widget key.

```python
st.text_area("Avoid list", key="avoid_list_text")  # not "avoid_list"
avoid = [s.strip() for s in st.session_state["avoid_list_text"].split(",")]
```

## initial_prompt must include all TasteProfile fields including principles

`profile.principles` is the most important part of the taste profile — the overarching synthesis. Omitting it from the prompt gives the generator only the dimension-level constraints.

Separate list items with `"; "` not `" "` so the model can parse them as distinct constraints.

```python
f"Taste principles: {'; '.join(profile.principles)}. "  # NOT " ".join
```

## Probe unknown API response formats before writing the consuming module

Before writing `generator/client.py`, run this and read the output:

```python
resp = raw_client.chat.completions.create(model=BASE_MODEL, messages=[...])
import json; print(json.dumps(resp.model_dump(), indent=2))
```

The correct parser follows from what you see, not what the docs say.

## .env must be in .gitignore

The `.env` file contains the `API_KEY`. Add it to `.gitignore` before any collaborators join:

```bash
echo '.env' >> .gitignore
```

## BASE_MODEL should be configurable via env var

Hardcoding `BASE_MODEL = "google/gemini-3.1-flash-image-preview"` creates a single point of failure. Use an env var with a default:

```python
BASE_MODEL = os.environ.get("MODEL", "google/gemini-3.1-flash-image-preview")
```
