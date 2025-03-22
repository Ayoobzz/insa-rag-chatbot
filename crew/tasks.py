from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List, Annotated
from langgraph.graph import add_messages
import logging
from datetime import datetime, timedelta


class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    next: str


def main_agent_node(state: AgentState, agents: dict) -> dict:
    agent = agents["Main_agent"]
    response = agent["llm"].invoke(state["messages"])
    logging.debug(f"Main_agent raw response: {response}")

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        query = tool_args.get("__arg1", state["messages"][-1].content)

        if tool_name == "RAG_Search":
            tool_output = agents["Main_agent"]["tools"][0].func(query)
            response = AIMessage(content=tool_output)
            next_step = END
        elif tool_name == "Timetable_Lookup":
            next_step = "Timetable_agent"
            response = AIMessage(content="Delegating to timetable specialist...")
        else:
            response = AIMessage(content="Unknown tool called.")
            next_step = END
    else:
        query = state["messages"][-1].content.lower()
        current_date = datetime(2025, 3, 22)  # Hardcoded per instructions
        if "date" in query or "today" in query:
            response = AIMessage(content=f"The current date is {current_date.strftime('%B %d, %Y')}.")
        elif "tomorrow" in query:
            tomorrow = current_date + timedelta(days=1)
            response = AIMessage(content=f"Tomorrow is {tomorrow.strftime('%B %d, %Y')}.")
        else:
            response_content = getattr(response, "content", "")
            if not response_content or "<tool-use>" in response_content:
                response = AIMessage(content="Hello! How can I assist you today?")
            else:
                response = AIMessage(content=response_content)
        next_step = END

    logging.debug(f"Main_agent processed response: {response.content}")
    return {
        "messages": [response],
        "next": next_step
    }


def timetable_agent_node(state: AgentState, agents: dict) -> dict:
    agent = agents["Timetable_agent"]
    response = agent["llm"].invoke(state["messages"])
    logging.debug(f"Timetable_agent raw response: {response}")

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        query = tool_call["args"].get("__arg1", state["messages"][-1].content)
        tool_output = agents["Timetable_agent"]["tools"][0].func(query)
        response = AIMessage(content=tool_output)
    else:
        query = state["messages"][-1].content.lower()
        response_content = getattr(response, "content", "")
        if not response_content or "<tool-use>" in response_content:
            response = AIMessage(content="No timetable information available.")
        else:
            response = AIMessage(content=response_content)

    logging.debug(f"Timetable_agent processed response: {response.content}")
    return {
        "messages": [response],
        "next": END
    }


def build_graph(agents: dict) -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("Main_agent", lambda state: main_agent_node(state, agents))
    graph.add_node("Timetable_agent", lambda state: timetable_agent_node(state, agents))
    graph.set_entry_point("Main_agent")
    graph.add_conditional_edges("Main_agent", lambda state: state["next"])
    graph.add_edge("Timetable_agent", END)
    return graph.compile()


def execute_task(query: str, agents: dict, past_messages: List[dict] = None) -> str:
    graph = build_graph(agents)
    if past_messages is None:
        past_messages = []
    messages = []
    for msg in past_messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["text"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["text"]))
    messages = messages[-5:]
    messages.append(HumanMessage(content=query))


    initial_state = {
        "messages": messages,
        "next": "Main_agent"
    }
    result = graph.invoke(initial_state)
    return result["messages"][-1].content