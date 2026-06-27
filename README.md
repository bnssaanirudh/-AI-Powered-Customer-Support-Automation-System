# AI-Powered Customer Support Automation System

This project is an AI-Powered Customer Support Automation System built using LangGraph, LangChain, and Groq. It automates the handling of customer support queries by routing them to specialized agents (Sales, Technical, Billing, Account), retrieving relevant context via a RAG pipeline, maintaining conversation memory, and escalating high-risk requests to a human supervisor.

## Features
- **Intent Classification**: Automatically categorizes customer queries.
- **Conditional Routing**: Directs queries to the correct department agent.
- **RAG Pipeline**: Retrieves relevant information from the company knowledge base (mock text documents).
- **SQLite Memory**: Maintains conversation history to answer follow-up queries. (`memory.db` is created automatically on first run; `schema.sql` is included).
- **Supervisor & Human-in-the-Loop**: Flags high-risk requests (e.g., refunds) and interrupts execution for human approval.

## Setup Instructions

1. **Install Python 3.9+**
2. **Install Dependencies**:
   Open your terminal and run:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you encounter issues with `pysqlite3`, standard `sqlite3` built into Python is used in `graph.py`.*

3. **Set Groq API Key**:
   Create a `.env` file in the root directory and add your API key, or set it directly in your environment:
   ```bash
   # On Windows
   set GROQ_API_KEY=your_api_key_here
   
   # On Mac/Linux
   export GROQ_API_KEY=your_api_key_here
   ```

## Run Instructions

Run the main demonstration script:
```bash
python main.py
```

### What to expect during execution:
The script provides an interactive terminal loop:
1. **Initialize Session**: You will be prompted to enter a `Customer ID`. This string is used as the SQLite `thread_id` to store conversation history natively across interactions.
2. **Interactive Chat**: Type your queries dynamically. The graph will route them and retrieve exact RAG context before answering.
3. **Supervisor HITL**: If you ask for a refund or account closure, the Supervisor node will intercept it, rewrite the response, flag it as HIGH RISK, and pause execution for your approval. **You must type 'y' or 'n' in the console.**
4. **Memory Verification**: Try asking about previous interactions to see the memory agent recall details dynamically.
A `workflow_diagram.png` will also be generated dynamically in the root directory showing the pure LangGraph architecture nodes.


## Submission Output
Once you have run the script, please take screenshots of the console output and generate the required PDF for submission. Then, bundle these project files into a ZIP file.
