from dotenv import load_dotenv
from src.utils.states import GenerateAnalystssState,InterviewState,ResearchGraphState
from src.utils.models  import llm 
from src.utils.objects import SearchQuery,Analyst,Perspectives
from src.utils.prompts import analyst_instructions,question_instructions,search_instructions,answer_instructions,section_writer_instructions,intro_conclusion_instructions,report_writer_instructions
load_dotenv()
from langchain.messages import SystemMessage,HumanMessage
from langgraph.types import interrupt
from langchain_tavily import TavilySearch
from langchain_core.messages import get_buffer_string
# nodes


def create_analysts(state:GenerateAnalystssState):
    '''create analysts'''
    topic =state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback =state.get("human_analyst_feedback")

    # Enforce structure output
    structure_llm =llm.with_structured_output(Perspectives)

    # SYSTEM _message
    system_message=analyst_instructions.format(topic=topic,human_analyst_feedback=human_analyst_feedback,
                                               max_analysts=max_analysts)

    analysts=structure_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="please generate the set of analysts.")])

    return {"analysts":analysts.analysts,
            "human_analyst_feedback":None}

def human_feedback(state:GenerateAnalystssState):
    """this is where the human gives feedback about the analysts given"""

    feedback = interrupt({
        "question": "Are these analysts okay for you?",
        "analysts": [
            analyst.model_dump() if hasattr(analyst, "model_dump") else analyst
            for analyst in state.get("analysts", [])

        
        ],
        "instuctions": "Return feedback to regenerate analysts "
        "or return empty/perfect/continue/okay to approve and continue the graph"
    })

    if feedback is None:
        return {"human_analyst_feedback": None}

    if isinstance(feedback, str):
        feedback = feedback.strip()

        if feedback == "":
            return {"human_analyst_feedback": None}

        if feedback.lower() in {"perfect", "okay", "continue", "yes"}:
            return {"human_analyst_feedback": None}

        return {"human_analyst_feedback":feedback}

    return {"human_analyst_feedback":None}

def generate_question(state: InterviewState):


    """node to generate the question"""

    # get state analyst
    analyst = state["analyst"]

    if isinstance(analyst, dict):
        analyst = Analyst.model_validate(analyst)

    messages = state["messages"]

    # generate question
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages+
                          [HumanMessage(content="Ask your next questtion.")])

    return {"messages": [question]}


def search_web(state:InterviewState):
    """Retrieve docs from the web"""

    #search query 
    structured_llm=llm.with_structured_output(SearchQuery)
    search_instruction_system_message=SystemMessage(content=search_instructions)
    tavily_search=TavilySearch(max_results=3)
    search_query =structured_llm.invoke([search_instruction_system_message]+state['messages']
                                            + [HumanMessage(content="Write the search query for the analyst's last question.")])

    data=tavily_search.invoke({"query":search_query.search_query})
    search_docs=data.get("results",data)

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"context":[formatted_search_docs]}



def search_web2(state:InterviewState):

    """Retrieve docs from the web"""

    #search query 
    structured_llm=llm.with_structured_output(SearchQuery)
    search_instruction_system_message=SystemMessage(content=search_instructions)
    tavily_search=TavilySearch(max_results=3)
    search_query =structured_llm.invoke([search_instruction_system_message]+state['messages']
                                            + [HumanMessage(content="Write the search query for the analyst's last question.")])

    data=tavily_search.invoke({"query":search_query.search_query})
    search_docs=data.get("results",data)

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"context":[formatted_search_docs]}

def generate_answer(state: InterviewState):

    """Node to answer a question"""

    # get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    if isinstance(analyst, dict):
        analyst = Analyst.model_validate(analyst)

    # answer question
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)] + messages + [HumanMessage(content="please answer the analyst's last question.")])

    # name the message as coming from the expert
    answer.name = "expert"

    # append to the state
    return {"messages": [answer]}

def save_interview(state:InterviewState):
    """save interviews"""

    messages=state["messages"]

    interview=get_buffer_string(messages)

    return {'interview':interview}


def write_section(state:InterviewState):


    """Node to answer a question"""


    interview =state['interview']
    context=state['context']
    analyst=state['analyst']

    if isinstance(analyst,dict):
        analyst=Analyst.model_validate(analyst)

    system_message=section_writer_instructions.format(focus=analyst.description)
    section=llm.invoke([SystemMessage(content=system_message)]+
                       [HumanMessage(content=f"use this source to write your section : {context}")])

    return {"sections":[section.content]}


def write_report(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content=f"Write a report based upon these memos.")])
    return {"content": report.content}


def write_introduction(state: ResearchGraphState):

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    intro = llm.invoke([SystemMessage(content=instructions)]+ [HumanMessage(content=f"Write the report introduction")])
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    conclusion = llm.invoke([SystemMessage(content=instructions)] + [HumanMessage(content="Write the report conclusion")])
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """Reduce step: combine the introduction, body and conclusion into the final report."""

    content = state["content"]
    sources = None

    if content.startswith("## Insights"):
        content = content[len("## Insights"):].strip()
        if "\n## Sources\n" in content:
            content, sources = content.split("\n## Sources\n", 1)

    final_report = (
        state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    )

    if sources is not None:
        final_report += "\n\n## Sources\n" + sources

    return {"final_report": final_report}