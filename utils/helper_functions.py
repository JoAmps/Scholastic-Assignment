from typing import List
from langchain_core.messages import BaseMessage
from data_validator.data_valid import AgentState


class InteractionHandler:
    def __init__(self, model_instance):
        """
        Initialize the handler with a bound model.
        """
        self.model = model_instance

    def run_llm(self, state: AgentState) -> AgentState:
        """
        Executes the LLM with the current message state.
        """
        messages = state.get("messages", [])
        response = self.model.invoke(messages)

        state["messages"] = [response]  # Replace with new assistant message
        state["final_answer"] = response.content
        state["original_llm_response"] = state.get(
            "original_llm_response", response.content
        )
        state["is_valid"] = True  # Confirm valid after LLM completes

        return state

    def get_feedback(self, state: AgentState) -> AgentState:
        """
        Prepares state for feedback after LLM completes.
        """
        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        state["messages"] = []
        state["final_answer"] = last_message.content if last_message else ""
        return state

    def handle_feedback(self, state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        last_human = next(
            (msg for msg in reversed(messages) if msg.type == "human"), None
        )

        # Optionally store actual feedback (for logging)
        if last_human:
            print("🔁 Feedback received:", last_human.content)

        # Reset so feedback path isn't re-triggered
        state["user_feedback"] = None

        # Clear stale response to allow re-generation
        state["final_answer"] = None
        state["is_valid"] = True  # Skip validation again

        return state
