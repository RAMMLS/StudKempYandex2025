import json
from typing import List

from Logger import logger

LLM_INTENT_PROMPT_TEMPLATE = """
Ты – помощник по безопасности.
Задача: получив пользовательский ввод и совпавшие шаблоны регулярных выражений, определить, является ли ввод безвредным (например, учебный пример, фрагмент кода, обычный вопрос), вредоносным (попытка извлечь данные, SQL-инъекция, XSS, удалённое выполнение кода) или неоднозначным.

Верни один JSON-объект с ключами:
-intent: одно из значений ["benign","malicious","ambiguous"]
-confidence: число с плавающей точкой от 0.0 до 1.0
-explanation: короткая причина на человеческом языке (максимум 80 символов)
-recommended_action: одно из ["allow","ask_clarification","escalate_human","block"]
-normalized_input: предобработанный текст (для аудита)

User_input: <<USER_INPUT>>
Matched_patterns: <<MATCHED_PATTERNS>>
Верни только JSON.
"""

def ask_intent_llm(yandex_bot, cleaned_text: str, matched_patterns: List[str]):
    prompt = LLM_INTENT_PROMPT_TEMPLATE.replace("<<USER_INPUT>>", cleaned_text)\
                                       .replace("<<MATCHED_PATTERNS>>", json.dumps(matched_patterns))
    raw = yandex_bot.ask_gpt(prompt)
    try:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        j = raw[start:end]
        result = json.loads(j)
        return result
    except Exception as e:
        logger.warning("Failed parse LLM output for intent: %s", e)
        return {
            "intent": "ambiguous",
            "confidence": 0.4,
            "explanation": "llm_response_parse_failed",
            "recommended_action": "ask_clarification",
            "normalized_input": cleaned_text
        }