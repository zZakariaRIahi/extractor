import json
import os
import time
import traceback
from typing import Any, Dict, List

import streamlit as st
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
from pydantic import BaseModel, ValidationError

from core.state import BidState
from prompts.extractor_prompts import get_extractor_prompt


class ExtractionResult(BaseModel):
    fileClassification: Dict[str, List[int]]
    projectInfo: Dict[str, Any]
    projectReport: str


def _get_client() -> genai.Client:
    # GenAI SDK will read env vars if present, but Streamlit secrets are not env vars.
    # So we pass api_key explicitly.
    api_key = (
        st.secrets.get("GEMINI_API_KEY")
        or st.secrets.get("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        st.error("Missing GEMINI_API_KEY / GOOGLE_API_KEY in Streamlit secrets.")
        st.stop()
    return genai.Client(api_key=api_key)


def _wait_until_active(client: genai.Client, f, timeout_s: int = 180):
    """Gemini uploaded files can be PROCESSING; only ACTIVE is usable for inference."""
    start = time.time()
    while True:
        fresh = client.files.get(name=f.name)
        state = getattr(getattr(fresh, "state", None), "name", None) or str(getattr(fresh, "state", None))
        if state == "ACTIVE":
            return fresh
        if state == "FAILED":
            raise RuntimeError(f"Gemini file processing FAILED for: {fresh.name}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"File not ACTIVE after {timeout_s}s. Current state: {state}")
        time.sleep(2)


def extractor_agent(state: BidState) -> BidState:
    client = _get_client()

    # Ensure files are ACTIVE + re-fetched under this client
    try:
        active_files = []
        for f in (state.gemini_files or []):
            if not f or not getattr(f, "name", None):
                continue
            active_files.append(_wait_until_active(client, f))
        state.gemini_files = active_files
    except Exception:
        state.errors.append("File activation error:\n" + traceback.format_exc())
        return state

    files = [f for f in (state.gemini_files or []) if getattr(f, "uri", None)]
    if not files:
        state.errors.append("Extractor error: no valid uploaded Gemini files (missing .uri or .name).")
        return state

    prompt = get_extractor_prompt()

    # More robust than passing File objects directly: build explicit URI Parts
    parts = [
        genai_types.Part.from_uri(
            file_uri=f.uri,
            mime_type=getattr(f, "mime_type", "application/pdf"),
        )
        for f in files
    ]
    contents = [prompt, *parts]

    try:
        response = client.models.generate_content(
            model=st.session_state.get("MODEL_NAME", "gemini-2.5-flash"),
            contents=contents,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,  # structured output
            ),
        )
    except genai_errors.APIError:
        state.errors.append("Extractor Gemini APIError:\n" + traceback.format_exc())
        return state
    except Exception:
        state.errors.append("Extractor Gemini call error:\n" + traceback.format_exc())
        return state

    # When response_schema is used, parsed is usually available
    extracted = None
    if getattr(response, "parsed", None) is not None:
        extracted = response.parsed
        state.raw_extractor_output = json.dumps(extracted.model_dump(), ensure_ascii=False)
    else:
        response_text = (response.text or "").strip()
        state.raw_extractor_output = response_text
        try:
            data = json.loads(response_text)
            extracted = ExtractionResult.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            state.errors.append(f"Extractor JSON parse/validate error: {e}")
            return state

    state.file_classification = extracted.fileClassification
    state.project_info = extracted.projectInfo
    state.project_report = extracted.projectReport

    fc = extracted.fileClassification or {}
    drawings_idxs = fc.get("drawingsFiles") or fc.get("drawings") or_
