from dotenv import load_dotenv
load_dotenv()
import re
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from state import CustomerSupportState
from rag_pipeline import retrieve_context

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Deterministic high-risk detection (regex) ────────────────────────────────
# These patterns guarantee HITL for assignment-required cases regardless of LLM output.
HIGH_RISK_PATTERNS = [
    r"\brefund(s|ed)?\b",
    r"\bcancel(lation|led|ing)?\b",
    r"\b(close|closure|deactivate|delete)\s+(my\s+)?account\b",
    r"\bcompensation\b",
    r"\bescalat(e|ion)\b",
    r"\bmanagement\b",
]

def requires_human_approval(query: str) -> bool:
    """Return True if the query matches any high-risk pattern."""
    q = query.lower().strip()
    return any(re.search(pattern, q) for pattern in HIGH_RISK_PATTERNS)


# ── Intent classification ─────────────────────────────────────────────────────

def classify_intent_node(state: CustomerSupportState):
    """Classify the customer query into a department or Memory recall."""
    query = state.get("customer_query", "")

    # Deterministic memory-recall detection before calling LLM
    memory_triggers = [
        "previous", "past", "what was my", "last issue", "earlier issue",
        "recall", "remember", "history", "before"
    ]
    if any(t in query.lower() for t in memory_triggers):
        return {"department": "Memory"}

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify the user query into ONE of these exact words: Sales, Technical, Billing, Account.\n"
         "Output ONLY the single word, nothing else.\n"
         "Sales: product information, subscription plans, pricing details.\n"
         "Technical: application errors, installation issues, upload problems, crashes, configuration.\n"
         "Billing: invoice requests, payment issues, refund requests.\n"
         "Account: password reset, profile updates, account activation/deactivation."),
        ("human", "{query}")
    ])

    chain = prompt | llm
    result = chain.invoke({"query": query})
    dept = result.content.strip()
    if dept not in ["Sales", "Technical", "Billing", "Account"]:
        dept = "Sales"
    return {"department": dept}


# ── Department agents (RAG-grounded) ─────────────────────────────────────────

def department_agent(state: CustomerSupportState, department_name: str):
    """Generic RAG-grounded department agent."""
    query = state.get("customer_query", "")
    context = retrieve_context(query)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are a helpful customer support agent for the {{department_name}} department at ABC Technologies.\n"
         "Answer the user's query STRICTLY and ONLY using the facts in the Context below.\n"
         "DO NOT invent or guess any prices, policies, limits, or features not explicitly stated in the text.\n"
         "If the context does not contain the exact answer, say you will escalate the issue.\n"
         "For refund or policy questions, include ALL relevant details: timeframes, "
         "prorating rules, and human approval requirements.\n\n"
         "Context:\n{context}"),
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


# ── Memory recall agent (deterministic) ──────────────────────────────────────

def memory_agent_node(state: CustomerSupportState):
    """
    Deterministic memory agent.
    Reads HumanMessage history from state and returns the most recent
    previous query (excluding the current memory-recall question itself).
    No LLM call needed — purely based on stored conversation state.
    """
    current_query = state.get("customer_query", "").strip().lower()
    previous_queries = []

    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            content = str(message.content).strip()
            # Exclude the current memory-recall question itself
            if content.lower() != current_query:
                previous_queries.append(content)

    if previous_queries:
        last_issue = previous_queries[-1]
        response = (
            f'Based on our conversation history, your previous support issue was: '
            f'"{last_issue}"'
        )
    else:
        response = (
            "I could not find an earlier support issue in this conversation session. "
            "Please ensure you are using the same Customer ID as your previous session."
        )

    return {
        "retrieved_context": "",        # No RAG for memory recall
        "proposed_response": response,
        "is_high_risk": False,
        "human_approved": False,
    }


# ── Supervisor node ───────────────────────────────────────────────────────────

def supervisor_node(state: CustomerSupportState):
    """
    Supervisor: verifies RAG grounding, improves tone, and applies
    deterministic high-risk detection. Memory recall bypasses grounding check.
    """
    proposed_response = state.get("proposed_response", "")
    query = state.get("customer_query", "")
    context = state.get("retrieved_context", "")
    department = state.get("department", "")

    # Memory recall: no grounding check, never high-risk
    if department == "Memory":
        return {
            "proposed_response": proposed_response,
            "is_high_risk": False,
            "human_approved": False,
        }

    # Deterministic HITL decision — always authoritative, LLM cannot override
    is_high_risk = requires_human_approval(query)

    # LLM-based grounding verification and tone improvement
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Customer Support Supervisor at ABC Technologies.\n\n"
         "Review the proposed response using ONLY the Retrieved Context below.\n\n"
         "Rules:\n"
         "1. Remove any statement NOT supported by the retrieved context.\n"
         "2. Preserve exact prices, limits, policies, and time periods from the context.\n"
         "3. Do NOT invent timelines, website links, phone numbers, or processing guarantees.\n"
         "4. Improve the professional tone and completeness of the response.\n"
         "5. Your output MUST be exactly in this format:\n"
         "HIGH_RISK: YES or NO\n"
         "IMPROVED_RESPONSE: <your final grounded response>\n\n"
         "Retrieved Context:\n{context}"),
        ("human", "User Query: {query}\nProposed Response: {response}")
    ])

    chain = prompt | llm
    result = chain.invoke({"query": query, "response": proposed_response, "context": context})
    content = result.content

    # Parse improved response from supervisor output
    improved_response = proposed_response
    if "IMPROVED_RESPONSE:" in content:
        improved_response = content.split("IMPROVED_RESPONSE:")[1].strip()
    elif "IMPROVED_RESPONSE" in content:
        improved_response = content.split("IMPROVED_RESPONSE")[1].strip()
        if improved_response.startswith(":"):
            improved_response = improved_response[1:].strip()

    return {
        "is_high_risk": is_high_risk,       # Deterministic result is final
        "proposed_response": improved_response,
        "human_approved": False,
    }
