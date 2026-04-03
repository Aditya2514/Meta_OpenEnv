"""
smoke_test.py
-------------
Smoke-test for Phase 1 + Phase 2.

Runs one full episode per scenario using a simple heuristic agent and:
  1. Prints per-step logs (existing Phase 1 behaviour).
  2. Prints the final grader score (new Phase 2 behaviour).

Run from the repo root:
    python src/envs/assignment_planner/smoke_test.py
"""

import sys
import os

# Ensure the repo root is on the path so `src` resolves correctly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.envs.assignment_planner import (
    Action,
    AssignmentPlannerEnv,
    grade,
    list_task_ids,
)
from src.envs.assignment_planner.models import State


# ---------------------------------------------------------------------------
# Heuristic agent (unchanged from Phase 1)
# ---------------------------------------------------------------------------

def greedy_agent(obs) -> Action:
    """
    Prioritise high-severity bugs near deadlines.
    Spend as many hours as possible to avoid thin-spreading.
    Ask for help on tasks with >= 5 h estimated effort.
    """
    open_tasks = [t for t in obs.tasks if t.status != "done"]
    if not open_tasks:
        return Action(task_id=0, hours=0.01, ask_for_help=False)

    def priority(t):
        sev_map = {"high": 3, "medium": 2, "low": 1, None: 0}
        type_map = {"bug": 3, "review": 2, "feature": 1}
        return (type_map[t.type], sev_map[t.severity], -t.deadline)

    chosen = sorted(open_tasks, key=priority, reverse=True)[0]
    hours = min(obs.hours_left_today, chosen.remaining_hours)
    ask_help = chosen.estimated_hours >= 5.0

    return Action(task_id=chosen.id, hours=hours, ask_for_help=ask_help)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> float:
    """Run one full episode and return the grader score."""
    env = AssignmentPlannerEnv(task_id=task_id)
    obs = env.reset()

    print(f"\n{'=' * 62}")
    print(f"  Scenario : {task_id}")
    print(
        f"  Tasks    : {len(obs.tasks)}  |  "
        f"Max days : {env.max_days}  |  "
        f"Daily cap : {env.daily_capacity}h"
    )
    print(f"{'=' * 62}")

    trajectory: list[State] = [env.state()]   # include initial state
    total_reward = 0.0
    step = 0
    done = False

    while not done:
        action = greedy_agent(obs)
        obs, reward, done, info = env.step(action)
        trajectory.append(env.state())         # record state after each step
        total_reward += reward
        step += 1

        task_name = env._tasks[action.task_id].name
        print(
            f"  step {step:>3}  | day {obs.day}  | "
            f"task={action.task_id} ('{task_name[:30]}')  | "
            f"hours={action.hours:.1f}  | "
            f"reward={reward:+.3f}  | "
            f"open={obs.summary.tasks_remaining}"
        )

    all_done = all(t.status == "done" for t in obs.tasks)
    grader_score = grade(task_id, trajectory)

    print(f"\n  ✅ Episode finished")
    print(f"     all_tasks_done  : {all_done}")
    print(f"     total_reward    : {total_reward:+.3f}")
    print(f"     grader_score    : {grader_score:.4f}  (0.0 – 1.0)")

    return grader_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scores = {}
    for tid in list_task_ids():
        scores[tid] = run_episode(tid)

    print(f"\n{'=' * 62}")
    print("  SUMMARY")
    print(f"{'=' * 62}")
    for tid, sc in scores.items():
        bar = "█" * int(sc * 20)
        print(f"  {tid:<12}  score={sc:.4f}  |{bar:<20}|")
    print()
