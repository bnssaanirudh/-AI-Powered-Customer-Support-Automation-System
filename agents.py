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
    """Memory recall agent: answers follow-up questions from conversation history stored in SQLite."""
    messages = state.get("messages", [])
    
    # Build a readable history string from stored HumanMessage / AIMessage pairs
    history_str = "\n".join(
        [f"{m.type.upper()}: {m.content}" for m in messages if m.type in ('human', 'ai')]
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a customer support agent. The user is asking about their previous support interactions. "
                   "Answer ONLY using the Conversation History below. Do NOT invent or assume anything. "
                   "State only what is clearly present in the history. Be concise and factual."),
        ("human", "Conversation History:\n{history}\n\nQuery: {query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"history": history_str, "query": state["customer_query"]})
    
    return {
        "proposed_response": response.content,
        "retrieved_context": ""  # Clear stale RAG context — memory uses history, not documents
    }

# Deterministic high-risk keyword detection — never missed by the LLM
HIGH_RISK_KEYWORDS = [
    "refund",
    "cancel",
    "cancellation",
    "close my account",
    "account closure",
    "compensation",
    "escalate",
    "escalation",
    "management",
]

def supervisor_node(state: CustomerSupportState):
    """Supervisor: verifies RAG grounding, improves tone, and detects high-risk requests."""
    proposed_response = state.get("proposed_response", "")
    query = state.get("customer_query", "")
    context = state.get("retrieved_context", "")
    
    # --- Deterministic high-risk check (always reliable) ---
    query_lower = query.lower()
    is_high_risk = any(keyword in query_lower for keyword in HIGH_RISK_KEYWORDS)
    
    # --- LLM-based grounding verification and tone improvement ---
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Customer Support Supervisor at ABC Technologies.\n\n"
         "Review the proposed response using ONLY the Retrieved Context and Conversation History below.\n\n"
         "Rules:\n"
         "1. Remove any statement NOT supported by the retrieved context or conversation history.\n"
         "2. Preserve exact prices, limits, policies, and time periods from the context.\n"
         "3. Do NOT invent timelines, website links, phone numbers, or processing guarantees.\n"
         "4. Improve the professional tone and completeness of the response.\n"
         "5. Your output MUST be exactly in this format (two lines only):\n"
         "HIGH_RISK: YES or NO\n"
         "IMPROVED_RESPONSE: <your final grounded response>"
         "\n\nRetrieved Context:\n{context}"),
        ("human", "User Query: {query}\nProposed Response: {response}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": query, "response": proposed_response, "context": context})
    
    content = result.content
    
    # Also check LLM's risk assessment (OR with deterministic check)
    if "HIGH_RISK: YES" in content.upper():
        is_high_risk = True
    
    # Parse the improved response
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
