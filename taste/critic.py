from __future__ import annotations
import base64
import io
import json

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
    dims_text = "\n".join(
        f"- {d}: {getattr(profile, d)}" for d in CRITIQUE_DIMENSIONS[:-1]
    )
    avoid_text = ", ".join(profile.avoid_list)
    principles_text = "\n".join(f"- {p}" for p in profile.principles)
    return f"""You are a taste critic. Score this generated image against a specific aesthetic taste profile.

TASTE PROFILE:
Overarching principles:
{principles_text}

Dimensions:
{dims_text}
- avoid_list: must avoid {avoid_text}

SCORING INSTRUCTIONS:
Score each dimension 0-10 (how well the image matches that taste dimension).
Give an overall score 0-10.
In one sentence, name the single biggest gap between the image and the taste.
Write an improved generation prompt that directly addresses the biggest gap.

Return a JSON object with exactly these keys:
- "score": float 0-10
- "breakdown": object with keys {json.dumps(CRITIQUE_DIMENSIONS)} each a float 0-10
- "reasoning": string (one sentence)
- "revised_prompt": string (improved prompt for next iteration)"""


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
