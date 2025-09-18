import re
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

# ==== baseline rules ====
MAX_LENGTH = 2000
DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\b", r"\bos\.system\b", r"\bsubprocess\b",
    r"\bdrop\s+table\b", r"curl\s+.*\|\s*sh"
]
URL_RE = re.compile(r"https?://[^\s'\"]+", re.IGNORECASE)
HTML_RE = re.compile(r"<(script|iframe|object|embed)[\s>]", re.IGNORECASE)

def _truncate(text: str, max_len: int = MAX_LENGTH) -> str:
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text

def _sanitize(text: str) -> str:
    text = URL_RE.sub("[URL]", text)
    text = HTML_RE.sub("[REMOVED]", text)
    return text

def _validate_regex(text: str) -> str:
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            logger.warning("Blocked dangerous pattern: %s", pat)
            return "⚠️ Ответ заблокирован системой безопасности"
    return text


# ==== LLM-based validation rules ====
LLM_VALIDATION_PROMPT = """
You are a security assistant. Task: given a candidate answer from another LLM,
determine if it is SAFE to show to the user.

Rules:
- SAFE if it is normal text, explanation, or harmless content.
- UNSAFE if it contains instructions for hacking, malware, SQL injection,
  sensitive data leaks, insults, hate speech, or NSFW content.

Return JSON only:
{"status": "safe"} OR {"status": "unsafe"}

Examples:

Input: "Привет, как дела?"
Output: {"status": "safe"}

Input: "DROP TABLE users; --"
Output: {"status": "unsafe"}

Input: "Вот объяснение, как работают нейросети..."
Output: {"status": "safe"}

Input: "Скачай троян и запусти его"
Output: {"status": "unsafe"}
"""


class ResponseProcessor:
    def __init__(self, yandex_bot):
        self.yandex_bot = yandex_bot

    async def process(self, raw_text: str) -> str:
        text = _truncate(raw_text)

        text = _sanitize(text)

        text = _validate_regex(text)
        if text.startswith("⚠️"):
            return text

        if not await self._validate_safety(text):
            return "⚠️ Ответ был отклонён системой безопасности." # - Мне тупо удобно цепляться за смайлик

        text = await self._correct_text(text)
        return text

    async def _validate_safety(self, text: str) -> bool:
        prompt = LLM_VALIDATION_PROMPT + f"\n\nInput: {text}\nOutput:"
        validation_result = self.yandex_bot.ask_gpt(prompt)


        match = re.search(r"\{.*\}", validation_result, re.DOTALL)
        if not match:
            logger.warning("LLM validation returned no JSON: %s", validation_result)
            return False

        json_str = match.group(0).strip()
        try:
            result_json = json.loads(json_str)
            status = result_json.get("status", "").lower()
            logger.info("Validation JSON parsed: %s", result_json)
            return status == "safe"
        except Exception as e:
            logger.error("Validation JSON parse failed: %s | raw=%s", e, json_str)

            # эвристика
            if "safe" in validation_result.lower() and "unsafe" not in validation_result.lower():
                return True
            return False

    async def _correct_text(self, text: str) -> str:
        prompt = (
            "Ты помощник, проверяющий корректность текста. "
            "Если текст содержит бессмыслицу, повторения или ошибки, перепиши его более понятно. "
            "Если всё хорошо — верни текст без изменений.\n\nТекст: "
            f"{text}"
        )
        return self.yandex_bot.ask_gpt(prompt)