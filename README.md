# AnyGen: A Unified Interface for Text Generation

`AnyGen` is a minimal Python library that unifies text generation tasks using Hugging Face, OpenAI, and Gemini models. It offers a minimalistic and unified pipeline for loading models and generating outputs with ease and efficiency.

## Features
- Support for Hugging Face models
- Support for OpenAI's GPT models
- Support for Gemini models
- Easy-to-use interface for text generation

## Installation
### Using `pip`
You can install `AnyGen` from PyPI:
```bash
pip install -U anygen
```

### From Source
Clone the repository and install it manually:
```bash
git clone https://github.com/macabdul9/AnyGen.git
cd AnyGen
pip install .
```

### Requirements
Ensure the following libraries are installed:
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
generated_text = hf_generator.generate(prompt)
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
generated_text = openai_generator.generate(prompt)
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
generated_text = gemini_generator.generate(prompt)
print(generated_text)
```

### Example with Parameters
```python
from anygen import AnyGen

# Initialize the generator
api_key_fp = "openai_keys.json"  # Example for OpenAI
openai_generator = AnyGen(model_type="openai", api_key_fp=api_key_fp)

# Generate text with parameters
prompt = "Write python code for binary search"
parameters = {"temperature": 0.7, "max_tokens": 512}
generated_text = openai_generator.generate(prompt, parameters)
print(generated_text)
```

## API Key File Format
Both OpenAI and Gemini models require an API key stored in a JSON file. Below is an example format:

`openai_keys.json`:
```json
{
    "openai_model_name": {
        "api_key": "your_openai_api_key",
        "endpoint": "your_endpoint"
    }
}
```
Replace `openai_model_name` with the OpenAI model name (e.g., `gpt-4o-mini`), `api_key` with your API key, and `your_endpoint` with the provided endpoint URL.

`gemini_keys.json`:
```json
{
    "gemini_model_name": {
        "api_key": "your_gemini_api_key"
    }
}
```
Replace `gemini_model_name` with the [Gemini](https://aistudio.google.com/) model name (e.g., `gemini-2.0-flash-exp`), `api_key` with your Gemini API key.

## Parameters
- `temperature`: Controls the randomness of the output. Higher values produce more random results.
- `max_tokens`: The maximum number of tokens to generate.
- `top_p`: The cumulative probability of the top tokens to sample from.
- `top_k`: The number of highest probability vocabulary tokens to keep for top-k-filtering.
- `beam_size`: The number of beams to use for beam search.

## Contributions
Feel free to submit issues and/or contribute to this repository!

## License
This project is licensed under the MIT License.


## Cite this Work

If you use `AnyGen` in your research or work, please cite it using the following BibTeX entry:

```bibtex
@software{anygen,
  author = {Abdul Waheed},
  title = {AnyGen: A Unified Interface for Text Generation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/macabdul9/AnyGen},
  doi = {https://doi.org/10.5281/zenodo.14533072}
}
```