import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated 
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START,END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

class state(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

llm = ChatGroq(model_name = "llama-3.1-8b-instant",api_key = groq_api_key)

def llm_agent(state:state):
    return {"messages":llm.invoke(state["messages"])}

def simplegraph():
    # intialize the graph
    graph = StateGraph(state)

    # adding the nodes
    graph.add_node("llm_agent",llm_agent)

    # adding the edges
    graph.add_edge(START,"llm_agent")
    graph.add_edge("llm_agent",END)
    
    # compile the graph
    graph_builder1 = graph.compile()
    return graph_builder1

graph_builder1 = simplegraph()


def add(a:int, b:int):
    """ Add two numbers """
    return a + b

tools = ([add])

llm_with_tools = llm.bind_tools([add])

def llm_agent(state:state):
    return {"messages":llm_with_tools.invoke(state["messages"])}

# create the graph

def toolgraph():
    # intialize the graph
    graph = StateGraph(state)

    def should_continue(state:state):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    # adding the nodes
    graph.add_node("llm_agent",llm_agent)
    graph.add_node("tools",ToolNode(tools))

    # adding the edges
    graph.add_edge(START,"llm_agent")
    graph.add_edge("tools","llm_agent")
    graph.add_conditional_edges(
        "llm_agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )


    # compile the graph
    graph_builder = graph.compile()
    return graph_builder

graph_builder = toolgraph()