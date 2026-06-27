from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from state import CustomerSupportState
from agents import (
    classify_intent_node, 
    sales_agent_node, 
    technical_agent_node, 
    billing_agent_node, 
    account_agent_node,
    memory_agent_node,
    supervisor_node
)
import sqlite3

def route_intent(state: CustomerSupportState):
    department = state.get("department")
    if department == "Sales":
        return "sales_agent"
    elif department == "Technical" or department == "Technical Support":
        return "technical_agent"
    elif department == "Billing":
        return "billing_agent"
    elif department == "Account":
        return "account_agent"
    elif department == "Memory":
        return "memory_agent"
    else:
        # Default fallback
        return "sales_agent"

def hitl_check(state: CustomerSupportState):
    is_high_risk = state.get("is_high_risk", False)
    if is_high_risk:
        # Route to a human review node which will be interrupted
        return "human_review"
    else:
        return "finalize_response"

def human_review_node(state: CustomerSupportState):
    # This node is the breakpoint for human in the loop.
    # The state will be updated externally by the human (e.g. human_approved: True/False)
    approved = state.get("human_approved")
    if approved is None:
        # Should not happen if correctly interrupted and resumed with update
        return {"final_response": "Pending human approval."}
    
    if approved:
        return {"final_response": "Supervisor Approved: " + state.get("proposed_response", "")}
    else:
        return {"final_response": "Your request has been reviewed and rejected by a supervisor."}

def finalize_response_node(state: CustomerSupportState):
    # Simply pass the proposed response to final response
    return {"final_response": state.get("proposed_response", "")}

def create_workflow():
    workflow = StateGraph(CustomerSupportState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("sales_agent", sales_agent_node)
    workflow.add_node("technical_agent", technical_agent_node)
    workflow.add_node("billing_agent", billing_agent_node)
    workflow.add_node("account_agent", account_agent_node)
    workflow.add_node("memory_agent", memory_agent_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("finalize_response", finalize_response_node)
    
    # Add edges
    workflow.add_edge(START, "classify_intent")
    
    # Conditional routing based on intent
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "sales_agent": "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent": "billing_agent",
            "account_agent": "account_agent",
            "memory_agent": "memory_agent"
        }
    )
    
    # All agents (including memory) route through supervisor for universal quality control
    workflow.add_edge("sales_agent", "supervisor")
    workflow.add_edge("technical_agent", "supervisor")
    workflow.add_edge("billing_agent", "supervisor")
    workflow.add_edge("account_agent", "supervisor")
    workflow.add_edge("memory_agent", "supervisor")  # Memory also validated by supervisor (Task 9)
    
    # Conditional routing after supervisor for high risk
    workflow.add_conditional_edges(
        "supervisor",
        hitl_check,
        {
            "human_review": "human_review",
            "finalize_response": "finalize_response"
        }
    )
    
    workflow.add_edge("human_review", END)
    workflow.add_edge("finalize_response", END)
    
    # Setup Memory
    conn = sqlite3.connect("memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # Compile workflow with interrupt before human_review
    app = workflow.compile(checkpointer=memory, interrupt_before=["human_review"])
    
    return app
