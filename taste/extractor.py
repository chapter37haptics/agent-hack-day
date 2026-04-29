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
5. Avoid-list: what does this collection deliberately reject? List 3-5 specific items.

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
