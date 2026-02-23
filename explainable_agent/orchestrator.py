from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

from rich.console import Console
from rich.panel import Panel

from .agent import ExplainableAgent
from .openai_client import OpenAICompatClient
from .schemas import OrchestratorRunTrace, SubTaskTrace
from .json_utils import parse_json_object_relaxed

console = Console()

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class TeamOrchestrator:
    """
    Manages a team of ExplainableAgents, breaking down a main task,
    delegating it to sub-agents, and synthesizing the final answer.
    """
    def __init__(self, client: OpenAICompatClient, agents: dict[str, tuple[str, ExplainableAgent]], verbose: bool = False):
        """
        :param agents: A dictionary mapping agent names to a tuple of (description, ExplainableAgent instance)
                       Example: {"researcher": ("Can search the web", agent1), "coder": ("Can write python code", agent2)}
        """
        self.client = client
        self.agents = agents
        self.verbose = verbose

    def _generate_delegation_plan(self, main_task: str, requested_model: str) -> list[dict[str, str]]:
        agents_info = ""
        for name, (desc, _) in self.agents.items():
            agents_info += f"- Name: '{name}', Description: '{desc}'\n"

        system_prompt = (
            "You are the Lead Orchestrator of a multi-agent AI system.\n"
            "Your job is to break down the user's main task into sequential subtasks and assign them to your available agents.\n"
            "Rules:\n"
            "1. Only use the agents provided in the Available Agents list.\n"
            "2. Make the 'assigned_task' as detailed as possible so the sub-agent knows exactly what to do.\n"
            "3. Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "plan": [\n'
            '    {"agent_name": "...", "assigned_task": "...", "rationale": "..."}\n'
            "  ]\n"
            "}\n"
        )
        user_prompt = f"Available Agents:\n{agents_info}\n\nMain Task:\n{main_task}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        resolved_model = self.client.resolve_model(requested_model)
        
        # Request JSON format specifically
        raw_output = self.client.get_alternative_answer(
            model=resolved_model,
            task=_messages_to_prompt(messages),
            temperature=0.0,
            reasoning_effort="high"
        )
        
        payload, _ = parse_json_object_relaxed(raw_output)
        if isinstance(payload, dict) and "plan" in payload and isinstance(payload["plan"], list):
            return payload["plan"]
            
        # Fallback if parsing fails entirely
        return []

    def _synthesize_final_answer(self, main_task: str, subtasks: list[SubTaskTrace], requested_model: str) -> str:
        synthesis_input = ""
        for st in subtasks:
            synthesis_input += f"--- Agent: {st.agent_name} ---\n"
            synthesis_input += f"Assigned Task: {st.assigned_task}\n"
            synthesis_input += f"Final Answer: {st.trace.final_answer}\n\n"
            
        prompt = (
            "You are the Lead Orchestrator of a multi-agent AI system. Your agents have completed their subtasks.\n"
            f"Main Task: {main_task}\n\n"
            "Agent Results:\n"
            f"{synthesis_input}\n"
            "Based on the results above, provide a comprehensive final synthesis that directly answers the Main Task. "
            "Do not output JSON, just write the final response clearly."
        )
        
        resolved_model = self.client.resolve_model(requested_model)
        return self.client.get_alternative_answer(
            model=resolved_model,
            task=prompt,
            temperature=0.2,
            reasoning_effort="high"
        )

    def _evaluate_orchestration(self, subtasks: list[SubTaskTrace]) -> list[str]:
        diagnostics = []
        if len(subtasks) == 0:
            diagnostics.append("WARNING: Orchestrator failed to generate a delegation plan or parse it correctly.")
            return diagnostics
            
        # Check if tasks were too complex for sub-agents (e.g., taking too many steps)
        for st in subtasks:
            if len(st.trace.steps) >= 5:
                diagnostics.append(f"Diagnostic: Sub-agent '{st.agent_name}' took {len(st.trace.steps)} steps to complete its task. The task ('{st.assigned_task[:50]}...') might have been too broad. Consider breaking it down further in the future.")
            
            error_steps = [s for s in st.trace.steps if s.decision.error_analysis]
            if error_steps:
                diagnostics.append(f"Diagnostic: Sub-agent '{st.agent_name}' encountered errors and had to self-heal {len(error_steps)} times during execution. This shows good resilience but may indicate vague instructions or failing external APIs.")
                
        # Check agent utilization
        used_agents = set(st.agent_name for st in subtasks)
        if len(used_agents) == 1 and len(self.agents) > 1:
            diagnostics.append(f"Diagnostic: The orchestrator only utilized one agent ('{list(used_agents)[0]}') despite having {len(self.agents)} available. Ensure the task actually requires a multi-agent setup.")
            
        return diagnostics

    def run(self, main_task: str, requested_model: str) -> OrchestratorRunTrace:
        run_id = "orch_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        started_at = _utc_now()
        
        if self.verbose:
            console.print(Panel(f"[bold white]Orchestrator Planning Phase[/bold white]\n[dim]Analyzing task and generating delegation plan...[/dim]", style="cyan"))
            
        plan = self._generate_delegation_plan(main_task, requested_model)
        
        if self.verbose and plan:
            plan_str = ""
            for i, step in enumerate(plan):
                plan_str += f"[bold yellow]{i+1}. {step.get('agent_name', 'Unknown')}[/bold yellow]: {step.get('assigned_task', '')}\n"
                plan_str += f"   [dim italic]Rationale: {step.get('rationale', '')}[/dim italic]\n"
            console.print(Panel(plan_str, title="Delegation Plan", style="green"))

        subtask_traces: list[SubTaskTrace] = []
        
        for step in plan:
            agent_name = step.get("agent_name", "")
            assigned_task = step.get("assigned_task", "")
            rationale = step.get("rationale", "")
            
            if agent_name not in self.agents:
                if self.verbose:
                    console.print(f"[red]Error: Orchestrator tried to use unknown agent '{agent_name}'. Skipping.[/red]")
                continue
                
            _, agent = self.agents[agent_name]
            
            if self.verbose:
                console.print(f"\n[bold magenta]>>> Handing over to Sub-Agent: {agent_name} <<<[/bold magenta]")
                
            # Run the sub-agent
            trace = agent.run(assigned_task)
            
            subtask_traces.append(SubTaskTrace(
                agent_name=agent_name,
                assigned_task=assigned_task,
                orchestrator_rationale=rationale,
                trace=trace
            ))
            
            if self.verbose:
                console.print(f"[bold magenta]<<< Sub-Agent {agent_name} finished. >>>[/bold magenta]\n")
                
        if self.verbose:
            console.print(Panel("[bold white]Orchestrator Synthesis Phase[/bold white]\n[dim]Combining results into final answer...[/dim]", style="cyan"))
            
        final_synthesis = self._synthesize_final_answer(main_task, subtask_traces, requested_model)
        
        finished_at = _utc_now()
        diagnostics = self._evaluate_orchestration(subtask_traces)
        
        return OrchestratorRunTrace(
            run_id=run_id,
            main_task=main_task,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            subtasks=subtask_traces,
            final_synthesis=final_synthesis,
            diagnostics=diagnostics
        )

def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}:\n{content}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)
