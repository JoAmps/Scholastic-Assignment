# graph_builder/build_graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from data_validator.data_valid import AgentState
from utils.helper_functions import InteractionHandler
from data_sources.create_tools import ToolExecutor
from data_validator.validate_query import Validator

interactions = InteractionHandler(ToolExecutor().get_model())
executor = ToolExecutor()
validator = Validator()


def build_graph_with_feedback():
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("validate_input", validator.validate_input)
    builder.add_node("llm", interactions.run_llm)
    builder.add_node("tools", executor.execute_tools)
    builder.add_node("final_llm", interactions.run_llm)
    builder.add_node("get_feedback", interactions.get_feedback)
    builder.add_node("handle_feedback", interactions.handle_feedback)

    builder.set_entry_point("validate_input")

    builder.add_conditional_edges(
        "validate_input",
        lambda state: state.get("is_valid", False),
        {
            True: "llm",
            False: END,
        },
    )

    builder.add_edge("llm", "tools")
    builder.add_edge("tools", "final_llm")
    builder.add_edge("final_llm", "get_feedback")

    builder.add_conditional_edges(
        "get_feedback",
        lambda state: bool(state.get("user_feedback")),
        {True: "handle_feedback", False: END},
    )

    builder.add_edge("handle_feedback", "llm")

    return builder.compile(checkpointer=MemorySaver())
