"""Пример работы с чатом через gigachain"""
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.gigachat import GigaChat

class GigaChatBot:
    def __init__(self, credentials, verify_ssl_certs=False, model='GigaChat'):
        # Авторизация в сервисе GigaChat
        self.chat = GigaChat(credentials=credentials, verify_ssl_certs=verify_ssl_certs, model=model)
        self.messages = [
            SystemMessage(
                content="Ты бот-программист, который помогает пользователю решить его задачи, а так же хорошо разбираешься во всех аспектах программирования и тестирования. Так же ты можешь помочь с теоретическими вопросами"
            )
        ]

    def get_response(self, user_input):
        # Ввод пользователя
        self.messages.append(HumanMessage(content=user_input))
        res = self.chat.invoke(self.messages)
        self.messages.append(res)
        # Ответ модели
        return res.content