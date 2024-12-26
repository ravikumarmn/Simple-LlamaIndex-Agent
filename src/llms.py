
from llama_index.llms.openai import OpenAI


def get_openai_model(model="gpt-3.5-turbo", system_prompt=""):
    """
    Creates and returns an OpenAI model object.

    Args:
        model (str): The OpenAI model version to use (default is "gpt-3.5-turbo").
        system_prompt (str): Optional system prompt for the model.

    Returns:
    OpenAI: A configured OpenAI model object.
    """
    return OpenAI(
        model=model,
        max_tokens=4_096,
        temperature=0.7,
        system_prompt=system_prompt,
    )
