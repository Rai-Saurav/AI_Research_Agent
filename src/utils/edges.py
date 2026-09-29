from dotenv import load_dotenv
# from src.utils.states import GenerateAnalystsState
from src.utils.states import GenerateAnalystssState,InterviewState,ResearchGraphState
from typing import Literal
from langgraph.graph import END
from langchain_core.messages import AIMessage,HumanMessage
from langgraph.types import Send

load_dotenv()

def should_continue(state: GenerateAnalystssState) -> Literal['create_analysts',END]:
    """Return the next node to exute"""

    human_analyst_feedback = state.get("human_analyst_feedback", None)

    if human_analyst_feedback:
        return "create_analysts"

    return END

def routes_messages(state:InterviewState,name:str="expert"):

    """Route between question and answer"""

    messages=state["messages"]
    max_num_turns=state.get("max_num_turns",2)

    num_responses=len([m for m in messages if isinstance(m,AIMessage) and m.name ==name])

    if num_responses >=max_num_turns:
        return "save_interview"

    return "ask_question"

# def initiate_all_interview(state:ResearchGraphState):
#     """this is the map step where we run each interview in sub graph using Send API"""

def initiate_all_interviews(state: ResearchGraphState):
    """Map step: run each interview in a subgraph using the Send API"""

    human_analyst_feedback = state.get("human_analyst_feedback")

    if human_analyst_feedback:
        return "create_analysts"

    topic = state["topic"]

    return [
        Send("conduct_interview",{
                "analyst": analyst,
                "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")],
            })for analyst in state["analysts"]
    ]