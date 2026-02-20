import os
from explainable_agent.config import Settings
from explainable_agent.openai_client import OpenAICompatClient
from explainable_agent.agent import ExplainableAgent
from explainable_agent.report import write_run_artifacts

def main():
    print("Initializing Explainable Agent...")
    
    # 1. Load settings (automatically reads from .env if present)
    # You can also pass overrides manually: Settings(base_url="...", requested_model="...")
    settings = Settings.from_env()
    
    # 2. Initialize the LLM client
    client = OpenAICompatClient(base_url=settings.base_url, api_key=settings.api_key)
    
    # 3. Create the agent instance. 
    # Set verbose=True to see the beautiful, colored step-by-step trace in your terminal!
    agent = ExplainableAgent(settings=settings, client=client, verbose=True)
    
    # 4. Define your task
    # This example asks the agent to use the built-in DuckDuckGo web search tool
    task = "duckduckgo_search: What are the latest updates in Python 3.13?"
    
    print(f"\n[Starting Task] {task}\n")
    
    # 5. Run the agent and get the execution trace
    try:
        trace = agent.run(task)
    except Exception as e:
        print(f"Agent failed to execute: {e}")
        return

    # 6. Save the run artifacts (traces and markdown reports)
    trace_path, report_path = write_run_artifacts(trace, settings.runs_dir)
    
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(trace.final_answer)
    print("="*50)
    
    print(f"\nSaved compact trace to: {trace_path}")
    print(f"Saved human-readable diagnostic report to: {report_path}")
    print("\nTip: Open the report.md file to see your agent's error breakdowns, hallucination risks, and diagnostic suggestions!")

if __name__ == "__main__":
    main()
