import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from graph_builder.build_graph import build_graph_with_feedback

# --- Session Initialization ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "validated" not in st.session_state:
    st.session_state.validated = False

if "history" not in st.session_state:
    st.session_state.history = [] 

# --- Build LangGraph ---
graph = build_graph_with_feedback()
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- UI Title ---
st.title("📍 Location Info Assistant")

# --- Input Box ---
user_query = st.text_input("Ask a location-based question (e.g., weather in Nairobi):")


def invoke_graph(messages, user_feedback=None):
    input_state = {"messages": messages}
    if user_feedback:
        input_state["user_feedback"] = user_feedback

    result = graph.invoke(input_state, config=config)

    if not user_feedback and not result.get("is_valid", True):
        st.session_state.validated = False
        st.warning(result.get("validation_error", "❌ Invalid input. Please rephrase."))
        return None

    st.session_state.validated = True
    st.session_state.messages += result.get("messages", [])

    final_response = result.get("final_answer", "")
    tools_used = result.get("tools_used", [])

    # ✅ Always track response + tools, even if empty
    st.session_state.history.append(
        {
            "response": final_response or "⚠️ No response generated.",
            "tools": tools_used or [],
        }
    )

    return final_response


# --- Submit Initial Query ---
if st.button("Submit") and user_query:
    st.session_state.messages = [HumanMessage(content=user_query)]
    invoke_graph(st.session_state.messages)

# --- Display Chat History (All interactions) ---
for idx, entry in enumerate(st.session_state.history):
    st.subheader(f"💡 Assistant Response {idx + 1}")
    st.write(entry["response"])
    if entry["tools"]:
        st.info("🔧 Data sources: " + ", ".join(entry["tools"]))

# --- Feedback Loop for Most Recent Entry ---
if st.session_state.history:
    with st.form("feedback_form", clear_on_submit=False):
        feedback = st.radio("Was this helpful?", ["👍", "👎"], key="feedback_radio")
        correction_input = ""
        if feedback == "👎":
            correction_input = st.text_input(
                "What would you like to clarify or change?", key="correction_input"
            )

        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            if feedback == "👍":
                st.success("✅ Thanks for your feedback!")
                st.stop()
            elif correction_input:
                correction_msg = HumanMessage(content=correction_input)
                st.session_state.messages.append(correction_msg)

                revised_response = invoke_graph(
                    messages=st.session_state.messages,
                    user_feedback=correction_input,  # ✅ <-- PASS the actual feedback
                )

                if revised_response:
                    st.rerun()
                else:
                    st.warning("⚠️ No revised response generated.")
            else:
                st.warning("⚠️ Please provide clarification.")
