import os
import subprocess
from pathlib import Path
from explainable_agent.config import Settings
from explainable_agent.agent import ExplainableAgent
from explainable_agent.openai_client import OpenAICompatClient
from explainable_agent.tools import define_tool
from explainable_agent.report import write_run_artifacts

# ==========================================
# 1. CUSTOM TOOL: SELF-HEALING DEMO
# ==========================================
@define_tool(
    name="get_user_email",
    description="Fetches the email address for a given user ID.",
    usage_hint="Input should be just the numeric user ID, e.g., 101."
)
def get_user_email(user_id_str: str, _: Path) -> str:
    user_db = {"101": "alice@example.com"}
    user_id_str = user_id_str.strip()
    email = user_db.get(user_id_str)
    
    if email:
        return f"The email for user ID {user_id_str} is {email}."
    else:
        # Returning an ERROR string triggers the agent's Self-Healing mode!
        return f"ERROR: User ID '{user_id_str}' not found in the database. Please try using a web search."

# ==========================================
# RUNNER FUNCTIONS
# ==========================================
def run_agent_scenario(title, task, settings, client):
    print("\n" + "="*60)
    print(f"🎬 SCENARIO: {title}")
    print("="*60)
    print(f"[Task] {task}\n")
    
    agent = ExplainableAgent(settings=settings, client=client, verbose=True)
    try:
        trace = agent.run(task)
        trace_path, report_path = write_run_artifacts(trace, settings.runs_dir)
        print(f"\n✅ Final Answer: {trace.final_answer}")
        print(f"📊 Report saved to: {report_path}")
    except Exception as e:
        print(f"❌ Scenario failed: {e}")

def run_evaluation_scenario():
    print("\n" + "="*60)
    print("🎬 SCENARIO: JSONL DATASET EVALUATION PIPELINE")
    print("="*60)
    print("The framework includes a powerful evaluation pipeline that can read .jsonl files,")
    print("test the model's tool-calling accuracy, repair broken JSONs, and score the arguments.")
    print("Running evaluation on 'examples/custom_eval_sample.jsonl'...\n")
    
    eval_script = Path("scripts/eval_hf_tool_calls.py").resolve()
    dataset_path = Path("examples/custom_eval_sample.jsonl").resolve()
    
    import sys
    # We use sys.executable to ensure we run the evaluation with the same Python environment
    cmd = [
        sys.executable, str(eval_script),
        "--dataset", str(dataset_path),
        "--base-url", "http://localhost:1234/v1",
        "--model", "qwen/qwen3-vl-4b" # Adjust this if using LM Studio (e.g. gpt-oss-20b)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        print("--- Evaluation Output ---")
        # Just print the last 20 lines to show the scores
        lines = result.stdout.strip().split('\n')
        for line in lines[-20:]:
            print(line)
        if result.stderr:
            print("\nErrors:", result.stderr)
    except Exception as e:
        print(f"Failed to run evaluation: {e}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("🚀 EXPLAINABLE AGENT LAB - FULL FEATURE SHOWCASE 🚀\n")

    settings = Settings.from_env().with_overrides(
        base_url="http://localhost:1234/v1", # Changed to 1234 for LM Studio
        requested_model="qwen/qwen3-vl-4b",
        reasoning_effort="high",
        max_steps=5
    )
    client = OpenAICompatClient(base_url=settings.base_url, api_key=settings.api_key)

    # SCENARIO 1: Built-in Tools (SQLite & Math)
    task1 = (
        "First, run 'sqlite_init_demo' to create a demo database. "
        "Then, use 'sqlite_query' to select the name and email of all customers from the customers table. "
    )
    run_agent_scenario("Built-in Tools (SQLite Database)", task1, settings, client)

    # SCENARIO 2: Custom Tools & Self-Healing
    task2 = (
        "I need the email for user ID 999 using get_user_email. "
        "If you get an error that the user doesn't exist, self-heal and use "
        "duckduckgo_search to search for 'Explainable AI Agent Framework' and summarize."
    )
    run_agent_scenario("Custom Tool & Self-Healing", task2, settings, client)

    # SCENARIO 3: Chaos Engineering (Stress Testing)
    task3 = (
        "Calculate (512 * 4) + 128 using the calculate_math tool."
    )
    chaos_settings = settings.with_overrides(chaos_mode=True)
    chaos_client = OpenAICompatClient(base_url=chaos_settings.base_url, api_key=chaos_settings.api_key)
    run_agent_scenario("Chaos Engineering Mode (20% Random Error Injection)", task3, chaos_settings, chaos_client)

    # SCENARIO 4: JSONL Evaluation Pipeline
    run_evaluation_scenario()

if __name__ == "__main__":
    main()
