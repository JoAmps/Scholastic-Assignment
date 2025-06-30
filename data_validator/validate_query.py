from llm_models.llm import LLMHandler
from data_validator.data_valid import AgentState


class Validator:
    """
    Validates user input to confirm it includes a valid location and topic.
    """

    def __init__(self):
        self.model = LLMHandler().location_validation_llm()

    def validate_input(self, state: AgentState) -> AgentState:
        """
        Validates the user's query unless it's feedback.
        Adds 'is_valid' and 'validation_error' fields to state.
        """

        # ✅ Skip validation for feedback loop
        if "user_feedback" in state:
            state["is_valid"] = True
            return state

        messages = state.get("messages", [])
        if not messages:
            state["is_valid"] = False
            state["validation_error"] = "❌ No message provided."
            return state

        user_message = messages[-1].content

        # Prompt to extract structured info
        prompt = f"""
        Extract the following from the input:

        1. location: The name of the city or place.
        2. info_about_location: A Wikipedia-ready query combining the location and what the user is asking 
        (e.g., "Accra culture", "Toronto overview").

        If either can't be confidently extracted, return:
        {{
            "location": "INVALID",
            "info_about_location": "INVALID"
        }}

        Input: "{user_message}"

        Output format:
        {{
            "location": str,
            "info_about_location": str
        }}
        """

        result = self.model.invoke(prompt)
        print(f"🧪 Validation result: {result}", flush=True)

        if result.location != "INVALID" and result.info_about_location != "INVALID":
            state["is_valid"] = True
        else:
            state["is_valid"] = False
            state["validation_error"] = (
                "❌ Could not extract a valid location or topic. Please rephrase."
            )

        return state
