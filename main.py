# def hello_world():
#     return "Hello World!"

from chatbot import Chatbot

bot = Chatbot()
# encoded = bot.encode_prompt("Hello, how are you?")
# print(encoded)
reply = bot.decode_reply([15496, 703, 345, 30])
print(reply)