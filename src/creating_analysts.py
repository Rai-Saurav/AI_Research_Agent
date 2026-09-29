from src.utils.nodes import create_analysts,human_feedback
from langgraph.graph import START , END ,StateGraph
from src.utils.states import GenerateAnalystssState
from dotenv import load_dotenv
from src.utils.edges import should_continue

load_dotenv()

# creating our graph
# GenerateAnalystsState state
# node eate_analysts
# edges no need / simple 

builder=StateGraph(GenerateAnalystssState)

builder.add_node("create_analysts",create_analysts)
builder.add_node("human_feedback",human_feedback)

builder.add_edge(START,"create_analysts")
builder.add_edge("create_analysts","human_feedback")
builder.add_conditional_edges("human_feedback",should_continue)

graph=builder.compile()