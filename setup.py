from setuptools import setup, find_packages

setup(
    name="anygen",
    version="1.0.2",
    author="Abdul Waheed",
    author_email="abdulwaheed1513@gmail.com",
    description="A unified interface for text generation using Hugging Face, OpenAI, and Gemini models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/macabdul9/AnyGen",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "transformers",
        "google-generativeai",
        "requests",
        "openai",
    ],
)
