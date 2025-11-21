from transformers import AutoTokenizer, AutoModelForCausalLM

class Chatbot:
    
    def __init__(self):
        model_name = "microsoft/DialoGPT-small"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def encode_prompt(self, prompt: str):
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return encoded
    
    def decode_reply(self, reply_ids: list[int]):
        decoded = self.tokenizer.decode(reply_ids, skip_special_tokens=True)
        return decoded
        