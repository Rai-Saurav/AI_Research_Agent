from typing_extensions import TypedDict,Annotated, NotRequired
from typing import Optional, List
from src.utils.objects import Analyst
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
import operator

# state
class GenerateAnalystssState(TypedDict):
    topic: str  # Research Topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: NotRequired[Optional[str]]  # Human feedback for what is
    analysts: NotRequired[List[Analyst]]=Field(description="Generate analysts as per the user says")  # List of all our Analysts

class InterviewState(MessagesState):
    # you AUTOMATICALLY get a "messages" field for free, with correct append behavior
    # then you just ADD whatever extra fields your interview needs on top:
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str #interview transcript
    sections: list # final key we duplicate in outer state for send() api

class ResearchGraphState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: NotRequired[Optional[str]]  # Human feedback
    analysts: List[Analyst]  # Analyst asking questions
    sections: Annotated[list, operator.add]  # Send() API key
    introduction: str  # Introduction for the final report
    content: str  # Content for the final report
    conclusion: str  # Conclusion for the final report
    final_report: str  # Final report


