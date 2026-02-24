
import sys
from pathlib import Path

def run_test(name: str, base_url: str, model: str, task: str, chaos: bool = False) -> bool:
    from explainable_agent import ExplainableAgent, Settings
    from explainable_agent.openai_client import OpenAICompatClient
    from explainable_agent.report import write_run_artifacts

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  Task: {task[:80]}...")
    print("="*60)

    settings = Settings.from_env().with_overrides(
        base_url=base_url,
        requested_model=model,
        max_steps=5,
        chaos_mode=chaos,
    )
    client = OpenAICompatClient(base_url=settings.base_url, api_key=settings.api_key)
    agent = ExplainableAgent(settings=settings, client=client, verbose=True)

    try:
        trace = agent.run(task)
        trace_path, report_path = write_run_artifacts(trace, settings.runs_dir)
        print(f"\n[OK] Final answer (first 200 chars): {trace.final_answer[:200]}...")
        print(f"     Trace: {trace_path}")
        print(f"     Report: {report_path}")
        return True
    except Exception as e:
        print(f"\n[FAIL] {e}")
        return False


def main():
    import explainable_agent
    print(f"explainable-agent version: {explainable_agent.__version__}")
    assert explainable_agent.__version__ == "0.1.4", f"Expected 0.1.4, got {explainable_agent.__version__}"

    results = []

    # --- OLLAMA (granite4:3b) ---
    results.append(("Ollama: Math", run_test(
        "Ollama granite4:3b - calculate_math",
        base_url="http://localhost:11434/v1",
        model="granite4:3b",
        task="Calculate (15 * 7) + 23 using the calculate_math tool. Give the final number.",
    )))

    results.append(("Ollama: SQLite", run_test(
        "Ollama granite4:3b - SQLite",
        base_url="http://localhost:11434/v1",
        model="granite4:3b",
        task="Run sqlite_init_demo to create a demo DB, then use sqlite_query to SELECT name FROM customers LIMIT 3.",
    )))

    results.append(("Ollama: Chaos", run_test(
        "Ollama granite4:3b - Chaos Mode",
        base_url="http://localhost:11434/v1",
        model="granite4:3b",
        task="Calculate 100 + 50 using calculate_math.",
        chaos=True,
    )))

    # --- LM STUDIO ---
    results.append(("LM Studio: Math", run_test(
        "LM Studio - calculate_math",
        base_url="http://localhost:1234/v1",
        model="qwen/qwen3-vl-4b",
        task="Calculate 42 * 17 using the calculate_math tool. Give the final number.",
    )))

    results.append(("LM Studio: SQLite", run_test(
        "LM Studio - SQLite",
        base_url="http://localhost:1234/v1",
        model="qwen/qwen3-vl-4b",
        task="Run sqlite_init_demo, then sqlite_query to SELECT email FROM customers LIMIT 2.",
    )))

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
