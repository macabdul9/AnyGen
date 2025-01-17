import json
import time
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import google.generativeai as genai

class AnyGen:
    def __init__(self, model_type, model_name_or_path=None, device="cpu", api_key_fp=None):
        
        self.model_type = model_type.lower()
        
        load_fn = getattr(self, f"_load_{self.model_type}_model")
        
        if self.model_type in ["huggingface", "hf"]:
            self.model = load_fn(model_name_or_path=model_name_or_path, device=device)
        else:
            self.model = load_fn(api_key_fp=api_key_fp)


    def _load_credentials(self, file_path):
        with open(file_path) as f:
            return json.load(f)
        
    def _load_huggingface_model(self, model_name_or_path, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(device).eval()
        return pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    def _load_hf_model(self, model_name_or_path, device):
        return self._load_huggingface_model(model_name_or_path, device)

    def _load_gemini_model(self, api_key_fp):
        
        credentials = self._load_credentials(api_key_fp)
        model_name = list(credentials.keys())[0]
        genai.configure(api_key=credentials[model_name]["api_key"])
        return genai.GenerativeModel(model_name)
    
    def _load_openai_model(self, api_key_fp):
        return self._load_credentials(api_key_fp) # this laods the api key

    def _generate_from_huggingface(self, prompt, parameters):
        
        max_new_tokens = parameters.get("max_tokens", 512)
        del parameters["max_tokens"]
        
        generated_text =  self.model(
            prompt,
            return_full_text=False, # return generated tokens only 
            max_new_tokens=max_new_tokens, 
            **parameters
            
        )[0]['generated_text']
        return generated_text
    
    def __generate_from_hf(self, prompt, parameters):
        return self._generate_from_huggingface(prompt=prompt, parameters=parameters)

    def _generate_from_openai(self, prompt, parameters):
        model_id = list(self.model.keys())[0]  # this returns the model name for which resource has been created
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}]
        }
        if parameters:
            payload.update({
                "temperature": parameters.get("temperature"),
                "max_tokens": parameters.get("max_tokens")
            })
        headers = {
            "Content-Type": "application/json",
            "api-key": self.model[model_id]["api_key"]
        }
        response = requests.post(self.model[model_id]['endpoint'], headers=headers, json=payload)
        
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def _generate_from_gemini(self, prompt, parameters):
        generation_config = {}
        if parameters:
            generation_config.update({
                "temperature": parameters.get("temperature"),
                "max_output_tokens": parameters.get("max_tokens")
            })
        response = self.model.generate_content(prompt, generation_config={k: v for k, v in generation_config.items() if v is not None})
        time.sleep(6)  # Avoid rate limiting
        return response.text

    def generate(self, prompt, parameters=None):
        
        # Dynamically determine the generation function based on model type
        generation_fn = getattr(self, f"_generate_from_{self.model_type}")
        
        return generation_fn(prompt=prompt, parameters=parameters)