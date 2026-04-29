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
# Use JSON mode — tokenrouter does not support tool_calls (Mode.TOOLS fails with 403)
instructor_client: instructor.Instructor = instructor.from_openai(
    _openai_client, mode=instructor.Mode.JSON
)

# Raw client for image generation and probing
raw_client: openai.OpenAI = _openai_client
