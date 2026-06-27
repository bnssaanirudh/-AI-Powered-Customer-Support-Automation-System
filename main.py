import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from graph import create_workflow

def draw_workflow(app):
    try:
        png_data = app.get_graph(xray=True).draw_mermaid_png()
        with open("workflow_diagram.png", "wb") as f:
            f.write(png_data)
        print("Workflow diagram saved to workflow_diagram.png")
    except Exception as e:
        print("Could not generate workflow diagram (requires graphviz/mermaid dependencies):", e)

def main():
    load_dotenv()
    
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY environment variable not set. Please set it to run the agents.")
        return

    print("Initializing LangGraph Application...")
    app = create_workflow()
    
    # Save the workflow diagram
    draw_workflow(app)
    
    customer_id = input("Enter Customer ID: ").strip()
    if not customer_id:
        customer_id = "default_customer"
    thread_id = f"customer_{customer_id}"
    config = {"configurable": {"thread_id": thread_id}}

    print("\nSystem ready! Type your queries below (or 'exit' to quit):")
    query_count = 1
    while True:
        q = input(f"\n[{query_count}] You: ").strip()
        if not q:
            continue
        if q.lower() in ('exit', 'quit', 'q'):
            break

        print(f"\n{'='*50}")
        print(f"PROCESSING QUERY: {q}")
        print(f"{'='*50}")

        # Pass HumanMessage so memory agent can read conversation history.
        # Reset all transient fields to prevent state bleed between queries.
        initial_state = {
            "customer_query": q,
            "messages": [HumanMessage(content=q)],
            "department": "",
            "retrieved_context": "",
            "proposed_response": "",
            "is_high_risk": False,
            "human_approved": False,
            "final_response": "",
        }
        
        # Execute workflow
        for event in app.stream(initial_state, config, stream_mode="values"):
            pass  # Iterate to reach the end or HITL breakpoint
        
        # Check current state
        state = app.get_state(config)
        
        if state.values.get("retrieved_context"):
            print("\nRETRIEVED CONTEXT:\n", state.values.get("retrieved_context"))
        
        # If interrupted by human_review node
        if state.next == ('human_review',):
            print("\n>>> HITL BREAKPOINT TRIGGERED <<<")
            print(f"Supervisor flagged this request as HIGH RISK.")
            print(f"Proposed Response: {state.values.get('proposed_response')}")
            
            # Simulate human review
            user_input = input("Approve this request? (y/n): ")
            human_approved = user_input.strip().lower() == 'y'
            
            print(f"\nResuming execution with human_approved = {human_approved}...")
            
            # Update the state with the human's decision and resume
            app.update_state(config, {"human_approved": human_approved})
            for event in app.stream(None, config, stream_mode="values"):
                pass
            
            # Fetch final state
            state = app.get_state(config)
        
        # Print final response and department
        print(f"\nROUTED TO: {state.values.get('department')}")
        print(f"FINAL RESPONSE: {state.values.get('final_response')}")
        
        # Save the AI response to message history so memory agent can recall it next turn
        if state.values.get("final_response"):
            app.update_state(config, {"messages": [AIMessage(content=state.values.get("final_response"))]})
            
        print("[System] Conversation state successfully saved to SQLite memory.db checkpoint.")
            
        query_count += 1

if __name__ == "__main__":
    main()
