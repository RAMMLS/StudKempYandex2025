from typing import Dict

class PromptTemplate:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    def build_messages(self, user_input: str):
        return [
            {"role": "system", "text": self.system_prompt},
            {"role": "user", "text": user_input}
        ]