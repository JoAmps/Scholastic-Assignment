from graph_builder.build_graph import build_graph_with_feedback
import uuid

# Initialize the graph
graph = build_graph_with_feedback()
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


if __name__ == "__main__":
    # Create a unique session ID for state tracking
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    # Initial state for the graph
    inputs = {}

    # Run the information assistant
    def run_info_session():
        _ = graph.invoke(inputs, config=config)

    run_info_session()
