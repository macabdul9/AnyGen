import json
import time
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import google.generativeai as genai

class AnyGen:
    def __init__(self, model_type, model_name_or_path=None, device="cpu", api_key_fp=None):
        self.model_type = model_type.lower()
        self.model = None

        if self.model_type in ["huggingface", "hf"]:
            self.model = self._load_huggingface_model(model_name_or_path, device)
        elif self.model_type == "openai":
            self.credentials = self._load_credentials(api_key_fp)
        elif self.model_type == "gemini":
            self.model = self._create_gemini_model(api_key_fp)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _load_huggingface_model(self, model_name_or_path, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(device).eval()
        return pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    def _load_credentials(self, file_path):
        with open(file_path) as f:
            return json.load(f)

    def _create_gemini_model(self, api_key_fp):
        credentials = self._load_credentials(api_key_fp)
        model_name = list(credentials.keys())[0]
        genai.configure(api_key=credentials[model_name]["api_key"])
        return genai.GenerativeModel(model_name)

    def generate(self, prompt, parameters=None):
        if self.model_type in ["huggingface", "hf"]:
            return self._generate_from_huggingface(prompt, parameters)
        elif self.model_type == "openai":
            return self._generate_from_openai(prompt, parameters)
        elif self.model_type == "gemini":
            return self._generate_from_gemini(prompt, parameters)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _generate_from_huggingface(self, prompt, parameters):
        kwargs = {}
        if parameters:
            kwargs.update({
                "return_full_text": False,
                "temperature": parameters.get("temperature"),
                "max_new_tokens": parameters.get("max_tokens")
            })
        return self.model(prompt, **{k: v for k, v in kwargs.items() if v is not None})[0]["generated_text"]

    def _generate_from_openai(self, prompt, parameters):
        model = list(self.credentials.keys())[0]
        payload = {
            "model": parameters.get("model", "gpt-4"),
            "messages": [{"role": "user", "content": prompt}]
        }
        if parameters:
            payload.update({
                "temperature": parameters.get("temperature"),
                "max_tokens": parameters.get("max_tokens")
            })
        headers = {
            "Content-Type": "application/json",
            "api-key": self.credentials[model]["api_key"]
        }
        response = requests.post(self.credentials[model]["endpoint"], headers=headers, json=payload)
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
