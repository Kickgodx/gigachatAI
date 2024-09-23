"""Пример работы с чатом через gigachain"""
from langchain.schema import HumanMessage, SystemMessage
# from langchain.chat_models.gigachat import GigaChat
from langchain_community.chat_models.gigachat import GigaChat
from gigachat_creds import auth_data, modelGigaChat

# Авторизация в сервисе GigaChat
chat = GigaChat(credentials=auth_data, verify_ssl_certs=False, model=modelGigaChat)

messages = [
    SystemMessage(
        content="Ты бот-программист, который помогает пользователю решить его задачи, а так же хорошо разбираешься во всех аспектах программирования и тестирования."
    )
]


while(True):
    # Ввод пользователя
    user_input = input("User: ")
    messages.append(HumanMessage(content=user_input))
    res = chat(messages)
    messages.append(res)
    # Ответ модели
    print("Bot: ", res.content)