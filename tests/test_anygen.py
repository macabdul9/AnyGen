import json
from anygen.anygen import AnyGen

def test_anygen():
    prompt = "Once upon a time, in a land far away"

    # Test Hugging Face Model
    # print("Testing Hugging Face Model...")
    # hf_model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with your Hugging Face model
    # hf_generator = AnyGen(model_type="huggingface", model_name_or_path=hf_model_name, device="cpu")
    # hf_output = hf_generator.generate(prompt, parameters=None)
    # print("Hugging Face Output:", hf_output)

    # Test OpenAI Model
    print("Testing OpenAI Model...")
    openai_api_key_fp = "openai_keys.json"  # Path to OpenAI API credentials
    openai_generator = AnyGen(model_type="openai", api_key_fp=openai_api_key_fp)
    openai_output = openai_generator.generate(prompt, parameters=None)
    print("OpenAI Output:", openai_output)

    # Test Gemini Model
    print("Testing Gemini Model...")
    gemini_api_key_fp = "gemini_keys.json"  # Path to Gemini API credentials
    gemini_generator = AnyGen(model_type="gemini", api_key_fp=gemini_api_key_fp)
    gemini_output = gemini_generator.generate(prompt, parameters=None)
    print("Gemini Output:", gemini_output)

if __name__ == "__main__":
    test_anygen()