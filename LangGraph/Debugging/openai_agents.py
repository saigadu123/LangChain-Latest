from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv

# Run langgraph dev in terminal to debug the graph

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

class State(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

llm = ChatOpenAI(temperature=0)

def make_default_graph():

    graph_workflow = StateGraph(State)
    def call_model(state):
        return {"messages":[llm.invoke(state["messages"])]}
    
    graph_workflow.add_node("agent",call_model)
    graph_workflow.add_edge(START,"agent")
    graph_workflow.add_edge("agent",END)

    agent = graph_workflow.compile()

    return agent

def make_alternative_graph():

    @tool
    def add(a:float,b:float):
        """ Add two numbers"""
        return a+b
    
    def multiply(a:float,b:float):
        """ Multiply two numbers"""
        return a*b

    tool_node = ToolNode([add,multiply])
    model_with_tools = llm.bind_tools([add,multiply])
    def call_model(state):
        return {"messages":[model_with_tools.invoke(state["messages"])]}
    
    def should_continue(state):
        if state["messages"][-1].tool_calls:
            return "tools"
        return END
    
    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent",call_model)
    graph_workflow.add_node("tools",tool_node)
    graph_workflow.add_edge(START,"agent")
    graph_workflow.add_conditional_edges(
        "agent",
        should_continue
    )
    graph_workflow.add_edge("tools","agent")
    agent = graph_workflow.compile()
    return agent


agent = make_alternative_graph()