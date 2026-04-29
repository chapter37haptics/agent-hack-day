from __future__ import annotations
import base64

from client import raw_client, BASE_MODEL


def generate(prompt: str) -> bytes:
    """Generate an image from a taste-encoded prompt. Returns raw PNG/JPEG bytes."""
    resp = raw_client.chat.completions.create(
        model=BASE_MODEL,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": f"Generate an image: {prompt}"}],
        }],
    )

    # Gemini image output via tokenrouter is in model_extra["images"], not content
    images = resp.choices[0].message.model_extra.get("images", [])
    if not images:
        raise ValueError(
            f"No image in response. content={resp.choices[0].message.content!r}, "
            f"model_extra keys={list(resp.choices[0].message.model_extra.keys())}"
        )

    data_url: str = images[0]["image_url"]["url"]
    if not data_url.startswith("data:"):
        raise ValueError(f"Expected base64 data URL, got: {data_url[:80]}")

    b64_data = data_url.split(",", 1)[1]
    return base64.b64decode(b64_data)
