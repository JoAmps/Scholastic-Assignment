from typing import TypedDict, Annotated, operator, Optional
from pydantic import BaseModel
from langchain_core.messages import AnyMessage
import warnings
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    original_llm_response: Optional[str]
    final_answer: Optional[str]
    validation_error: Optional[str]
    is_valid: Optional[bool]
    user_feedback: Optional[str]
    tools_used: Optional[list[str]]


class WeatherDetails(BaseModel):
    location: str
    info_about_location: str


class WikiInput(BaseModel):
    topic: str = Field(..., description="Topic to search on Wikipedia")


class WeatherInput(BaseModel):
    city: str = Field(..., description="City name to fetch current weather")


class NewsInput(BaseModel):
    query: str = Field(..., description="Topic or keyword to search news")
