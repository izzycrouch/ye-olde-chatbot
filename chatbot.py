from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Chatbot:
    
    def __init__(self):
        model_name = "microsoft/DialoGPT-small"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.chat_history_ids = None

        self.system_prompt = "You are a helpful assistant. Respond to the end of this conversation accordingly.\n"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        if self.device == "cuda":
            print("\nChatpot is using GPU acceleration.\n")
    
        else:
            print("\nChatpot is not using GPU acceleration.\n")
    

    def encode_prompt(self, prompt: str):
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return encoded
    
    def decode_reply(self, reply_ids: list[int]):
        decoded = self.tokenizer.decode(reply_ids, skip_special_tokens=True)
        return decoded
    
    def generate_reply(self, prompt: str):
        new_prompt = prompt + '\n'

        encoded_prompt = self.encode_prompt(new_prompt)
        
        prompt_ids = encoded_prompt['input_ids']
        prompt_attention_mask = encoded_prompt['attention_mask']

        if self.chat_history_ids == None:    
            encoded_system_prompt = self.encode_prompt(self.system_prompt)
            system_prompt_ids = encoded_system_prompt['input_ids']
            
            input_ids = torch.cat((system_prompt_ids, prompt_ids), dim=1).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
        
        else:
            input_ids = torch.cat((self.chat_history_ids, prompt_ids), dim=1).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)

        generate = self.model.generate(input_ids=input_ids, attention_mask=attention_mask , pad_token_id=self.tokenizer.eos_token_id, do_sample=True)
        
        token_ids = generate[0].tolist()
        list_input_ids = input_ids[0].tolist()
        input_len = len(list_input_ids)
        reply_ids = token_ids[input_len:]
        decoded_reply = self.decode_reply(reply_ids)
        
        self.chat_history_ids = generate
    
        return decoded_reply

    def reset_history(self):
        self.chat_history_ids = None
        return self.chat_history_ids