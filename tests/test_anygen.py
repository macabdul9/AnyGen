import json
from anygen.anygen import AnyGen

def test_anygen():
    prompt = "Write a Python function to calculate the factorial of a number."

    # Test Hugging Face Model
    print("Testing Hugging Face Model...")
    hf_model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with your Hugging Face model
    device = "cuda"  # Use "cpu" if you don't have a GPU
    hf_generator = AnyGen(model_type="huggingface", model_name_or_path=hf_model_name, device=device)
    hf_output = hf_generator.generate(prompt, parameters={"max_tokens": 1024})
    print("Hugging Face Output:", hf_output)

    # Test OpenAI Model
    print("Testing OpenAI Model...")
    openai_api_key_fp = "keys/openai_keys.json"  # Path to OpenAI API credentials
    openai_generator = AnyGen(model_type="openai", api_key_fp=openai_api_key_fp)
    openai_output = openai_generator.generate(prompt, parameters={"max_tokens": 512, "temperature": 0.7})
    print("OpenAI Output:", openai_output)

    # Test Gemini Model
    print("Testing Gemini Model...")
    gemini_api_key_fp = "keys/gemini_keys.json"  # Path to Gemini API credentials
    gemini_generator = AnyGen(model_type="gemini", api_key_fp=gemini_api_key_fp)
    gemini_output = gemini_generator.generate(prompt, parameters={"max_tokens": 512, "temperature": 0.7})
    print("Gemini Output:", gemini_output)


if __name__ == "__main__":
    test_anygen()
