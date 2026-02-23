import os
import warnings
from pathlib import Path

# Suppress ResourceWarnings from sqlite3/httpx in multi-agent runs (harmless, noisy)
warnings.filterwarnings("ignore", category=ResourceWarning)
from explainable_agent.config import Settings
from explainable_agent.openai_client import OpenAICompatClient
from explainable_agent.agent import ExplainableAgent
from explainable_agent.orchestrator import TeamOrchestrator
from explainable_agent.report import write_orchestrator_artifacts

def main():
    print("🤖 EXPLAINABLE AGENT LAB - MULTI-AGENT ORCHESTRATION DEMO 🤖\n")

    settings = Settings.from_env().with_overrides(
        base_url="http://localhost:1234/v1",  # Change to 11434 for Ollama
        requested_model="qwen/qwen3-vl-4b",    # Adjust for your model
        reasoning_effort="high",
        max_steps=5
    )
    client = OpenAICompatClient(base_url=settings.base_url, api_key=settings.api_key)

    # 1. Create specialized agents
    research_agent = ExplainableAgent(settings=settings, client=client, verbose=False)
    database_agent = ExplainableAgent(settings=settings, client=client, verbose=False)

    # 2. Define the team
    agents_team = {
        "researcher": (
            "Can search the web using duckduckgo_search, read texts, and summarize information.",
            research_agent
        ),
        "db_expert": (
            "Can run SQLite commands (sqlite_init_demo, sqlite_query, sqlite_execute) to create databases, tables, and insert or query data.",
            database_agent
        )
    }

    # 3. Create the Orchestrator
    orchestrator = TeamOrchestrator(client=client, agents=agents_team, verbose=True)

    main_task = (
        "Search the web for the top 3 biggest AI news from this week. "
        "Then, create a SQLite demo database and save these news into a table called 'ai_news' with columns 'title' and 'summary'."
    )
    
    print(f"\n[Main Task] {main_task}\n")
    
    try:
        # Run the Multi-Agent Team
        trace = orchestrator.run(main_task=main_task, requested_model=settings.requested_model)
        
        # Save the orchestration report
        trace_path, report_path = write_orchestrator_artifacts(trace, settings.runs_dir)
        
        print("\n" + "="*50)
        print("FINAL SYNTHESIS:")
        print("="*50)
        print(trace.final_synthesis)
        print("="*50)
        
        print(f"\nSaved orchestrator trace to: {trace_path}")
        print(f"Saved human-readable orchestration report to: {report_path}")
        print("\nTip: Open the orch_report.md file to see the meeting minutes, subtask breakdowns, and delegation diagnostics!")
        
    except Exception as e:
        print(f"Orchestration failed: {e}")

if __name__ == "__main__":
    main()
