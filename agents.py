from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from state import CustomerSupportState
from rag_pipeline import retrieve_context

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)



def classify_intent_node(state: CustomerSupportState):
    query = state.get("customer_query", "")
    
    # Check if the query is asking about past memory/issues
    if "previous" in query.lower() or "past" in query.lower() or "what was my" in query.lower():
         return {"department": "Memory"}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the user query into ONE of these exact words: Sales, Technical, Billing, Account. Output ONLY the word, nothing else. "
                   "Sales: Product information, subscription plans, pricing details. "
                   "Technical: Application errors, installation issues, login problems, configuration issues. "
                   "Billing: Invoice requests, payment issues, refund requests. "
                   "Account: Password reset, profile updates, account activation/deactivation."),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": query})
    
    dept = result.content.strip()
    if dept not in ["Sales", "Technical", "Billing", "Account"]:
        dept = "Sales"
    
    return {"department": dept}

def department_agent(state: CustomerSupportState, department_name: str):
    query = state.get("customer_query", "")
    
    # Retrieve context from RAG
    context = retrieve_context(query)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a helpful customer support agent for the {{department_name}} department at ABC Technologies. "
                   "Answer the user's query STRICTLY and ONLY using the facts provided in the Context below. "
                   "DO NOT invent or guess any prices, policies, limits, or features that are not explicitly stated in the text. "
                   "If the context does not contain the exact answer, you MUST state that you do not know but will escalate it. "
                   "For policy or refund questions, always provide the complete details mentioned in the text (like timeframes, human approval, or prorating) if relevant."
                   "\n\nContext:\n{{context}}"),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"department_name": department_name, "context": context, "query": query})
    
    return {
        "retrieved_context": context,
        "proposed_response": response.content
    }

def sales_agent_node(state: CustomerSupportState):
    return department_agent(state, "Sales")

def technical_agent_node(state: CustomerSupportState):
    return department_agent(state, "Technical Support")

def billing_agent_node(state: CustomerSupportState):
    return department_agent(state, "Billing")

def account_agent_node(state: CustomerSupportState):
    return department_agent(state, "Account")

def memory_agent_node(state: CustomerSupportState):
    # Retrieve answer using stored memory
    messages = state.get("messages", [])
    
    # We pass the conversation history to the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a customer support agent. The user is asking about their previous interactions. Answer their question using the conversation history provided below. Be concise."),
        ("human", "Conversation History:\n{history}\n\nQuery: {query}")
    ])
    
    history_str = "\n".join([f"{m.type}: {m.content}" for m in messages if m.type in ('human', 'ai')])
    chain = prompt | llm
    response = chain.invoke({"history": history_str, "query": state["customer_query"]})
    
    return {
        "proposed_response": response.content,
        "is_high_risk": False, # Memory recall is not high risk
        "retrieved_context": "" # Clear any stale RAG context so it doesn't show in debug
    }

def supervisor_node(state: CustomerSupportState):
    proposed_response = state.get("proposed_response", "")
    query = state.get("customer_query", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Customer Support Supervisor. Your job is to validate and improve the proposed response to the customer for tone and professionalism. "
                   "Also, determine if the user's request is HIGH RISK. High risk includes: Refund, Cancellation, Account closure, Compensation. "
                   "Your output must be exactly in this format:\n"
                   "HIGH_RISK: YES or NO\n"
                   "IMPROVED_RESPONSE: <your improved response>"),
        ("human", "User Query: {query}\nProposed Response: {response}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": query, "response": proposed_response})
    
    content = result.content
    is_high_risk = "HIGH_RISK: YES" in content.upper()
    
    improved_response = proposed_response
    if "IMPROVED_RESPONSE:" in content:
        improved_response = content.split("IMPROVED_RESPONSE:")[1].strip()
    elif "IMPROVED_RESPONSE" in content:
        improved_response = content.split("IMPROVED_RESPONSE")[1].strip()
        if improved_response.startswith(":"):
            improved_response = improved_response[1:].strip()
    
    return {
        "is_high_risk": is_high_risk,
        "proposed_response": improved_response
    }
