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

# test generate_reply method:
prompts = [
    "What's your name?",
    "What do you think about AI?",
    "Sorry, tell me your name again."
]
for prompt in prompts:
    reply = bot.generate_reply(prompt)
    print(f"Prompt: {prompt}")
    print(f"Reply: {reply}\n")

# def use_chatbot():
#     bot = Chatbot()
#     print("\nBot: Welcome to Chatbot!\n")
    
#     user = input("Enter your name: ")
#     question = input(f"\nBot: Hello {user}, how can Chatbot help? ")
    
#     while True:
#         reply = bot.generate_reply(question)
#         print(f"\n{user}: {question}")
#         print(f"Bot: {reply}\n")

#         another_question = input(f"\nBot: Can I help you with anything else (y/n)?  ")
    
#         if another_question != 'y':
#             print(f"Bot: Goodbye!\n")
#             bot.reset_history()
#             break
            
#         question = input(f"\nBot: How can I help?  ")
        
# use_chatbot()