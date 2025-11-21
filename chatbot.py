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
    
    def generate_reply(self, prompt: str):
        new_prompt = prompt + '\n'

        encoded_prompt = self.encode_prompt(new_prompt)
        
        prompt_ids = encoded_prompt['input_ids']
        list_prompt_ids = prompt_ids[0].tolist()
        prompt_len = len(list_prompt_ids)
    
        prompt_attention_mask = encoded_prompt['attention_mask']
        
        # generate = self.model.generate(input_ids=prompt_ids, attention_mask=prompt_attention_mask , pad_token_id=self.tokenizer.eos_token_id)
        generate = self.model.generate(input_ids=prompt_ids, attention_mask=prompt_attention_mask , pad_token_id=self.tokenizer.eos_token_id, do_sample=True, top_p=0.8)

        list_ids = generate[0].tolist()
        
        reply_ids = list_ids[prompt_len:]
        
        decoded_reply = self.decode_reply(reply_ids)
        
        return decoded_reply

