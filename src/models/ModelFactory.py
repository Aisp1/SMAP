from models.Gemini import Gemini
from models.OpenAI import ChatGPT
from models.OpenAI import GPT4
from models.OpenAI import DeepSeek
from models.OpenAI import Llama_8B


class ModelFactory:
    @staticmethod
    def get_model_class(model_name):
        if model_name == "Gemini":
            return Gemini
        elif model_name == "ChatGPT":
            return ChatGPT
        elif model_name == "DeepSeek":
            return DeepSeek
        elif model_name == "Llama_8B":
            return Llama_8B
        elif model_name == "GPT4":
            return GPT4
        else:
            raise Exception(f"Unknown model name {model_name}")
