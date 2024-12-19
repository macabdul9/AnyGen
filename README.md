# AnyGen: A Unified Interface for Text Generation

`AnyGen` is a minimal Python library that unifies text generation tasks using Hugging Face, OpenAI, and Gemini models. It offers a minimalistic and unified pipeline for loading models and generating outputs with ease and efficiency.


## Features
- Support for Hugging Face models
- Support for OpenAI's GPT models
- Support for Gemini models
- Easy-to-use interface for text generation

## Installation
Ensure you have the required libraries installed:
```bash
pip install transformers google-generativeai requests openai
```

## Usage
Below are step-by-step instructions to generate text using each model type.

### 1. Hugging Face Model
```python
from anygen import AnyGen

# Initialize the generator
model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with your Hugging Face model name
device = "cuda"  # Use "cpu" if GPU is not available
hf_generator = AnyGen(model_type="huggingface", model_name_or_path=model_name_or_path, device=device)

# Generate text
prompt = "Write python code for binary search"
parameters = {"temperature": 0.7, "max_tokens": 100}
generated_text = hf_generator.generate(prompt, parameters)
print(generated_text)
```

### 2. OpenAI Model
```python
from anygen import AnyGen

# Initialize the generator
api_key_fp = "openai_keys.json"  # Path to your OpenAI credentials file
openai_generator = AnyGen(model_type="openai", api_key_fp=api_key_fp)

# Generate text
prompt = "Write python code for binary search"
parameters = {"temperature": 0.7, "max_tokens": 100, "model": "gpt-4"}
generated_text = openai_generator.generate(prompt, parameters)
print(generated_text)
```

### 3. Gemini Model
```python
from anygen import AnyGen

# Initialize the generator
api_key_fp = "gemini_keys.json"  # Path to your Gemini credentials file
gemini_generator = AnyGen(model_type="gemini", api_key_fp=api_key_fp)

# Generate text
prompt = "Write python code for binary search"
parameters = {"temperature": 0.7, "max_tokens": 100}
generated_text = gemini_generator.generate(prompt, parameters)
print(generated_text)
```

## API Key File Format
Both OpenAI and Gemini models require an API key stored in a JSON file. Below is an example format:

`openai_keys.json`:
```json
{
    "gpt-4o-miini": {
        "api_key": "your-openai-api-key",
        "endpoint": "your_endpoint"
    }
}
```

`gemini_keys.json`:
```json
{
    "gemini-model-name": {
        "api_key": "your-gemini-api-key"
    }
}
```

## Parameters
- `temperature`: Controls the randomness of the output. Higher values produce more random results.
- `max_tokens`: The maximum number of tokens to generate.
- `model`: (OpenAI-specific) Specifies the model to use (e.g., `gpt-4`).

## Contributions
Feel free to submit issues or contribute to this repository!

## License
This project is licensed under the MIT License.
