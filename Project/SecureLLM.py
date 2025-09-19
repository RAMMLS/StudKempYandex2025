import json
class SecureLLM:
    def __init__(self, bot, LLM_INTENT_PROMPT_TEMPLATE, LLM_VALIDATION_PROMPT, SYSTEM_PROMPT):
        self.bot = bot
        self.LLM_INTENT_PROMPT_TEMPLATE = LLM_INTENT_PROMPT_TEMPLATE
        self.LLM_VALIDATION_PROMPT = LLM_VALIDATION_PROMPT
        self.SYSTEM_PROMPT = SYSTEM_PROMPT

    def validate_input(self, user_input, matched_patterns):
        text = self.LLM_INTENT_PROMPT_TEMPLATE \
            .replace("<<USER_INPUT>>", user_input) \
            .replace("<<MATCHED_PATTERNS>>", json.dumps(matched_patterns))

        messages = [
            #{"role": "system", "text": "Ты – помощник по безопасности."},
            {"role": "system", "text": text}
        ]
        return self.bot.ask_gpt(messages)

    def validate_output(self, candidate_answer):
        text = self.LLM_VALIDATION_PROMPT.replace(
            "Input: ", f"Input: {candidate_answer}\n"
        )

        messages = [
            #{"role": "system", "text": "You are a security assistant."},
            {"role": "system", "text": text}
        ]
        return self.bot.ask_gpt(messages)

    def generate_answer(self, user_input, context=None):
        messages = [
            {"role": "system", "text": self.SYSTEM_PROMPT},
            {"role": "user", "text": f"Вопрос: {user_input}\nКонтекст: {context or ''}"}
        ]
        return self.bot.ask_gpt(messages)