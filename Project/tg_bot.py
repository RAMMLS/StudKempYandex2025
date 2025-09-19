import requests
import tempfile
from langchain_community.document_loaders import PyPDFLoader


from telegram.ext import Application, MessageHandler, filters


class TelegramBotHandler:
    def __init__(self, token, message_callback):
        self.token = token
        self.message_callback = message_callback
        self.app = Application.builder().token(self.token).build()

        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

    async def _on_message(self, update, context):
        user_message = update.message.text
        user_id = update.message.chat_id
        reply = await self.message_callback(user_message)
        await update.message.reply_text(reply)

    def run(self):
        print("🚀 Telegram бот запущен")
        self.app.run_polling()


