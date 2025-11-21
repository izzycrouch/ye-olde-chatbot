# def hello_world():
#     return "Hello World!"

from chatbot import Chatbot

bot = Chatbot()

# test encode_prompt method:
# encoded = bot.encode_prompt("Hello, how are you?")
# print(encoded)

# test decode_reply method:
# reply = bot.decode_reply([15496, 703, 345, 30])
# print(reply)


prompt = "What is the weather like today?"
reply = bot.generate_reply(prompt)
print(f"Prompt: {prompt}")
print(f"Reply: {reply}")