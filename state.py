from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
import operator

class CustomerSupportState(TypedDict):
    # Customer and query details
    messages: Annotated[List[BaseMessage], operator.add]
    customer_query: str
    
    # Intent classification
    department: str # "Sales", "Technical", "Billing", "Account", "Memory"
    
    # RAG Context
    retrieved_context: str
    
    # Human-in-the-loop and Supervisor
    proposed_response: str
    is_high_risk: bool
    human_approved: bool
    
    # Final Response
    final_response: str
