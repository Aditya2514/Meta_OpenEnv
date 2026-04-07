import os
import sys
import json
import logging
import argparse
from typing import List, Optional

# Place this before other imports to ensure stdout is clean
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from openai import OpenAI
from src.envs.assignment_planner.environment import AssignmentPlannerEnv
from src.envs.assignment_planner.graders import grade
from src.envs.assignment_planner.models import Action, Observation, State
from src.envs.assignment_planner.task_config import list_task_ids

# --- Mandatory Environment Variables ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = os.getenv("BENCHMARK", "assignment-planner")

# Logging to stderr only
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

# --- Mandatory Loggers ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- Logic ---

def get_llm_action(client: OpenAI, obs: Observation, task_id: str) -> Action:
    # Heuristic fallback if LLM fails or for simple tasks
    # (Replace this with your prompt logic from your previous script if needed)
    open_tasks = [t for t in obs.tasks if t.status != "done"]
    if not open_tasks:
        return Action(task_id=0, hours=0.0, ask_for_help=False)
    
    # Simple Greedy logic as baseline
    chosen = sorted(open_tasks, key=lambda t: (t.type == "bug", t.severity == "high", -t.deadline), reverse=True)[0]
    hours = min(obs.hours_left_today, chosen.remaining_hours)
    return Action(task_id=chosen.id, hours=round(hours, 2), ask_for_help=(chosen.severity == "high"))

def run_task(task_id: str, client: OpenAI):
    env = AssignmentPlannerEnv(task_id=task_id)
    
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    trajectory = []
    
    try:
        raw_obs = env.reset()
        obs = Observation(**raw_obs.model_dump())
        trajectory.append(env.state())
        
        done = False
        while not done and steps_taken < 20: # Safety cap
            action = get_llm_action(client, obs, task_id)
            
            # Action string for logs
            action_str = f"work(task={action.task_id}, hours={action.hours})"
            
            raw_obs, reward, done, info = env.step(action)
            obs = Observation(**raw_obs.model_dump())
            trajectory.append(env.state())
            
            steps_taken += 1
            rewards.append(reward)
            
            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=None)
            
        score = grade(task_id, trajectory)
        success = score >= 0.1
        
    except Exception as e:
        score = 0.0
        success = False
        # Optional: print(f"[DEBUG] Error: {e}", file=sys.stderr)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Default tasks to run for the submission
    tasks_to_run = ["easy_1", "medium_1", "hard_1"]
    
    for tid in tasks_to_run:
        run_task(tid, client)

if __name__ == "__main__":
    main()