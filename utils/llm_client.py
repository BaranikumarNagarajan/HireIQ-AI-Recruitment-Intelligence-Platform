"""LLM client — supports Groq (primary), Claude (Anthropic), and xAI."""
import json
import logging
import requests
from config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY, CLAUDE_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    XAI_API_KEY, XAI_MODEL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def extract_json_text(text: str) -> str | None:
    """Extract the first valid JSON object or array from text."""
    cleaned = text.strip()
    if not cleaned:
        return None

    for start_idx, char in enumerate(cleaned):
        if char not in '{[':
            continue

        closing = '}' if char == '{' else ']'
        stack = [char]
        in_string = False
        escape = False

        for i in range(start_idx + 1, len(cleaned)):
            c = cleaned[i]
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == char:
                stack.append(c)
            elif c == closing:
                stack.pop()
                if not stack:
                    candidate = cleaned[start_idx:i + 1].strip()
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
    return None


def parse_llm_json_response(response: str):
    """Parse LLM response into JSON, extracting valid JSON from surrounding text if needed."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        json_text = extract_json_text(response)
        if json_text is not None:
            return json.loads(json_text)
        raise


# ---------------------------------------------------------------------------
# Claude / Anthropic provider
# ---------------------------------------------------------------------------

def is_claude_available() -> bool:
    """Return True if ANTHROPIC_API_KEY is configured."""
    return LLM_PROVIDER == "claude" and bool(ANTHROPIC_API_KEY)


def get_claude_response(prompt: str, temperature: float = 0.2) -> str:
    """Call Claude API via the Anthropic SDK."""
    if not ANTHROPIC_API_KEY:
        return "Error: ANTHROPIC_API_KEY not set in .env"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as exc:
        logger.error("Claude request failed: %s", exc)
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------

def is_groq_available() -> bool:
    """Return True if GROQ_API_KEY is configured."""
    return LLM_PROVIDER == "groq" and bool(GROQ_API_KEY)


def get_groq_response(prompt: str, temperature: float = 0.2) -> str:
    """Call Groq API (OpenAI-compatible, very fast)."""
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not set in .env"
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        chat = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
        )
        return chat.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Groq request failed: %s", exc)
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# xAI / Grok provider
# ---------------------------------------------------------------------------

def is_xai_available() -> bool:
    """Return True if XAI_API_KEY is configured."""
    return LLM_PROVIDER == "xai" and bool(XAI_API_KEY)


def get_xai_response(prompt: str, temperature: float = 0.2) -> str:
    """Call xAI API (OpenAI-compatible endpoint)."""
    if not XAI_API_KEY:
        return "Error: XAI_API_KEY not set in .env"
    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": XAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 1024,
            },
            timeout=60,
        )
        if not resp.ok:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            logger.error("xAI %d error for model '%s': %s", resp.status_code, XAI_MODEL, body)
            return f"Error: xAI {resp.status_code} — {body}"
        return resp.json()["choices"][0]["message"]["content"] or ""
    except Exception as exc:
        logger.error("xAI request failed: %s", exc)
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def get_llm_response(prompt: str, temperature: float = 0.2) -> str:
    """Get text response from the configured LLM provider."""
    if LLM_PROVIDER == "groq":
        return get_groq_response(prompt, temperature)
    if LLM_PROVIDER == "claude":
        return get_claude_response(prompt, temperature)
    if LLM_PROVIDER == "xai":
        return get_xai_response(prompt, temperature)
    logger.error("Unsupported LLM provider: %s", LLM_PROVIDER)
    return f"Error: Unsupported LLM provider '{LLM_PROVIDER}'. Set LLM_PROVIDER to 'groq', 'claude', or 'xai'."
