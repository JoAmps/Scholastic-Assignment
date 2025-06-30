from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from data_sources.wikipedia import fetch_wikipedia_information
from data_sources.weather import get_current_weather
from data_sources.news import fetch_news_articles
from data_validator.data_valid import AgentState, WikiInput, WeatherInput, NewsInput
from llm_models.llm import LLMHandler

# ----------------------- #
# 🎯 Tool Definitions     #
# ----------------------- #


@tool(args_schema=WikiInput)
def wikipedia_search(topic: str) -> str:
    """🔎 Searches Wikipedia for a summary of the given topic."""
    return fetch_wikipedia_information(topic)


@tool(args_schema=WeatherInput)
def weather_tool(city: str) -> str:
    """🌤️ Fetches current weather data for the specified city."""
    return get_current_weather(city)


@tool(args_schema=NewsInput)
def news_tool(query: str) -> str:
    """📰 Retrieves recent news articles based on the provided query."""
    return fetch_news_articles(query)


# -------------------------- #
# 🔧 Tool Executor Class     #
# -------------------------- #


class ToolExecutor:
    """
    Manages tool execution and tool-aware LLM for LangGraph agents.
    """

    def __init__(self):
        self.tools = [wikipedia_search, weather_tool, news_tool]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.model = LLMHandler().get_llm().bind_tools(self.tools)

    def get_model(self):
        """
        Return the tool-enabled LLM instance for LangGraph use.
        """
        return self.model

    def execute_tools(self, state: AgentState) -> AgentState:
        """
        Executes tool calls from the latest assistant message in the state.

        Args:
            state (AgentState): Current LangGraph state.

        Returns:
            AgentState: Updated state with tool responses and tools used.
        """
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])
        results = []
        tools_used = []

        for call in tool_calls:
            tool_name = call["name"]
            tool_id = call["id"]
            args = call["args"]

            if tool_name not in self.tools_by_name:
                output = "Tool not found."
            else:
                output = self.tools_by_name[tool_name].invoke(args)
                tools_used.append(tool_name)

            results.append(
                ToolMessage(
                    tool_call_id=tool_id,
                    name=tool_name,
                    content=str(output),
                )
            )

        state["messages"] = results
        state["tools_used"] = tools_used
        return state

    def tool_needed(self, state: AgentState) -> bool:
        """
        Checks if tool calls are present in the latest assistant message.

        Args:
            state (AgentState): Current LangGraph state.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        last_msg = state["messages"][-1]
        return hasattr(last_msg, "tool_calls") and bool(last_msg.tool_calls)
