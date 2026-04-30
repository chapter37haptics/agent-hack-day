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
