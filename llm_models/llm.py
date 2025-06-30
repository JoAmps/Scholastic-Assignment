from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from data_validator.data_valid import WeatherDetails
import os

# Load environment variables
load_dotenv()


class LLMHandler:
    """
    Handles creation and configuration of LLM instances using OpenAI models via LangChain.

    Supports:
    - Basic LLM instance for general chat/completion tasks.
    - Structured output LLM for extracting specific fields (e.g. location and info).
    """

    def __init__(self):
        """
        Initializes the LLMHandler with environment-based OpenAI credentials.

        Attributes:
            api_key (str): OpenAI API key from environment variable 'openai_api_key'.
            model_name (str): OpenAI model name from environment variable 'model_name'.
        """
        self.api_key = os.getenv("openai_api_key")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key in environment variables.")

    def get_llm(self, model_name="gpt-4.1-2025-04-14") -> ChatOpenAI:
        """
        Returns a basic LLM instance configured with the API key and model name.

        Returns:
            ChatOpenAI: Configured OpenAI LLM instance.
        """
        return ChatOpenAI(model_name=model_name, openai_api_key=self.api_key)

    def location_validation_llm(self):
        """
        Returns an LLM instance configured to extract structured location-related data
        from user input.

        This LLM is used to validate and extract:
            - `location`: The geographic place (e.g., city or region).
            - `info_about_location`: A refined query describing what the user wants to know about the location
            (e.g., "Accra culture", "Toronto weather").

        The response is structured using the WeatherDetails Pydantic model.

        Returns:
            ChatOpenAI: A language model instance with structured output for location validation.
        """
        location_llm = self.get_llm(
            model_name="gpt-4.1-mini-2025-04-14"
        ).with_structured_output(WeatherDetails)
        return location_llm
