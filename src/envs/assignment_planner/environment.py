"""
environment.py
--------------
Core OpenEnv-style environment for the Assignment & Bug-Fix Planner Agent.

The agent (student / junior developer) must decide:
  - which task to work on (task_id),
  - how many hours to spend (hours),
  - and whether to ask for help (ask_for_help),
across multiple days, subject to a daily working-hour capacity and
hard deadlines, while avoiding burnout and missed deadlines.

Phase 2 wires in task_config.py for fixed episode configs and graders.py
for final 0–1 scoring. Phase 3 adds HTTP / OpenEnv scaffolding.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

from .models import Action, Observation, State, Summary, Task
from .task_config import TASK_CONFIGS, get_config


class AssignmentPlannerEnv:
    """
    OpenEnv-style environment for the Assignment & Bug-Fix Planner scenario.

    The environment follows a gym-like interface:
        obs  = env.reset()
        obs, reward, done, info = env.step(action)
        s    = env.state()

    Parameters
    ----------
    task_id : str
        One of the scenario keys in task_config.TASK_CONFIGS (e.g. "easy_1").
    daily_capacity : float, optional
        Override the scenario's default daily working-hour capacity.
        If None, the value from task_config is used.
    max_days : int, optional
        Override the scenario's maximum episode length.
        If None, the value from task_config is used.
    """

    def __init__(
        self,
        task_id: str = "medium_1",
        daily_capacity: Optional[float] = None,
        max_days: Optional[int] = None,
    ):
        # Validate and load the canonical config for this scenario
        cfg = get_config(task_id)  # raises KeyError for unknown task_id

        self.task_id = task_id
        # Caller overrides take precedence; fall back to scenario defaults
        self.daily_capacity: float = daily_capacity if daily_capacity is not None else float(cfg["daily_capacity"])
        self.max_days: int = max_days if max_days is not None else int(cfg["max_days"])

        # Keep a reference to the raw task definitions for reset()
        self._task_defs: List[Dict[str, Any]] = cfg["tasks"]

        # Internal mutable state (initialised via reset())
        self._day: int = 0
        self._hours_left_today: float = self.daily_capacity
        self._tasks: List[Task] = []

        # Seed the state so the env is usable even before reset() is called
        self.reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Resets the environment to its initial configuration and returns the
        first Observation.

        All tasks are restored to their original estimated_hours with status
        "not_started".  The day counter and daily hours are reset to 0 and
        daily_capacity respectively.

        Returns
        -------
        Observation
            The initial observation of the episode.
        """
        self._day = 0
        self._hours_left_today = self.daily_capacity

        # Deep-copy the task definitions so mutations don't bleed across episodes.
        # task_config entries do NOT include runtime fields (remaining_hours /
        # status), so we inject fresh defaults here.
        raw_tasks = copy.deepcopy(self._task_defs)
        self._tasks = [
            Task(
                **t,
                remaining_hours=float(t["estimated_hours"]),
                status="not_started",
            )
            for t in raw_tasks
        ]

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Advances the environment by one agent decision.

        Parameters
        ----------
        action : Action
            The agent's chosen task_id, hours to spend, and help flag.

        Returns
        -------
        observation : Observation
            The environment state after applying the action.
        reward : float
            Dense, shaped reward signal for this step.
        done : bool
            True if the episode has ended (deadline exceeded or all tasks done).
        info : dict
            Auxiliary diagnostic information (useful for debugging/logging).
        """
        # ── Capture pre-step state snapshot for reward computation ──────────
        old_state = self.state()

        # ── Validate action.task_id ─────────────────────────────────────────
        n_tasks = len(self._tasks)
        assert 0 <= action.task_id < n_tasks, (
            f"action.task_id={action.task_id} is out of range "
            f"[0, {n_tasks - 1}]."
        )

        task = self._tasks[action.task_id]
        assert task.status != "done", (
            f"Task {action.task_id} ('{task.name}') is already done. "
            "Choose a different task."
        )

        # ── Clamp hours to what is actually available ───────────────────────
        max_spendable = min(self._hours_left_today, task.remaining_hours)
        hours = min(action.hours, max_spendable)
        hours = max(hours, 0.0)  # safety floor

        assert hours > 0, (
            "No hours are available to spend (hours_left_today=0 or task is done). "
            "The episode should already be done."
        )

        # ── Apply work to the chosen task ───────────────────────────────────
        task.remaining_hours = round(task.remaining_hours - hours, 4)
        self._hours_left_today = round(self._hours_left_today - hours, 4)

        if task.status == "not_started":
            task.status = "in_progress"
        if task.remaining_hours <= 0.0:
            task.remaining_hours = 0.0
            task.status = "done"

        # ── Advance day if daily capacity is exhausted ──────────────────────
        day_advanced = False
        if self._hours_left_today <= 0.0:
            self._day += 1
            day_advanced = True
            # Decrement deadlines for all unfinished tasks
            for t in self._tasks:
                if t.status != "done":
                    t.deadline = max(0, t.deadline - 1)
            # Replenish hours for the new day
            self._hours_left_today = self.daily_capacity

        # ── Determine episode termination ───────────────────────────────────
        all_done = all(t.status == "done" for t in self._tasks)
        deadline_expired = self._day >= self.max_days
        done = all_done or deadline_expired

        # If done, zero out remaining hours so the agent cannot act further
        if done:
            self._hours_left_today = 0.0

        # ── Compute dense reward ────────────────────────────────────────────
        new_state = self.state()
        reward = self._compute_step_reward(
            action=action,
            clamped_hours=hours,
            task=task,
            old_state=old_state,
            new_state=new_state,
            day_advanced=day_advanced,
            all_done=all_done,
            deadline_expired=deadline_expired,
        )

        # ── Build info dict ─────────────────────────────────────────────────
        info: Dict = {
            "clamped_hours": hours,
            "day_advanced": day_advanced,
            "all_tasks_done": all_done,
            "deadline_expired": deadline_expired,
        }

        return self._build_observation(), reward, done, info

    def state(self) -> State:
        """
        Returns the current full internal state (for debugging / serialisation).

        Returns
        -------
        State
            Snapshot of day, hours_left_today, and the complete task list.
        """
        return State(
            day=self._day,
            hours_left_today=self._hours_left_today,
            tasks=copy.deepcopy(self._tasks),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_summary(self) -> Summary:
        """Compute aggregated summary statistics from the current task list."""
        remaining_tasks = [t for t in self._tasks if t.status != "done"]
        high_bugs = [
            t
            for t in remaining_tasks
            if t.type == "bug" and t.severity == "high"
        ]
        min_deadline = (
            min(t.deadline for t in remaining_tasks) if remaining_tasks else 0
        )
        return Summary(
            tasks_remaining=len(remaining_tasks),
            high_severity_bugs_remaining=len(high_bugs),
            days_until_deadline=min_deadline,
        )

    def _build_observation(self) -> Observation:
        """Package current internal state into an Observation."""
        return Observation(
            day=self._day,
            hours_left_today=self._hours_left_today,
            tasks=copy.deepcopy(self._tasks),
            summary=self._build_summary(),
        )

    def _compute_step_reward(
        self,
        action: Action,
        clamped_hours: float,
        task: Task,
        old_state: State,
        new_state: State,
        day_advanced: bool,
        all_done: bool,
        deadline_expired: bool,
    ) -> float:
        """
        Dense, shaped reward for a single step.

        Design goals
        ------------
        1. Encourage working on high-severity bugs and near-deadline tasks.
        2. Penalise ignoring urgent work in favour of low-priority tasks.
        3. Penalise burnout (spending excessive hours / overloading a day).
        4. Give a small bonus for asking for help on genuinely hard tasks.
        5. Give a large terminal bonus for finishing all tasks on time.
        6. Give a large terminal penalty for missing the overall deadline.

        Returns
        -------
        float
            Reward signal (can be negative; not bounded to 0–1).
        """
        reward = 0.0

        # ── 1. Priority / urgency bonus (Enhanced High-Variance) ─────────────
        # High-severity bugs that are near their deadline yield massively non-linear bonuses.
        urgency_multiplier = 1.0
        if task.type == "bug" and task.severity == "high":
            urgency_multiplier += 3.0          # +3.0 for critical bugs (was 1.5)
        elif task.type == "bug" and task.severity == "medium":
            urgency_multiplier += 1.5
        
        # Exponential scaling for near-deadline
        if task.deadline <= 1:
            urgency_multiplier *= 3.0          # massive multiplier for 1 day left
        elif task.deadline <= 2:
            urgency_multiplier *= 2.0          
        elif task.deadline <= 4:
            urgency_multiplier *= 1.2

        # Base reward: quadratic completion fraction to reward deep work over shallow work
        fraction_done_this_step = (
            clamped_hours / task.estimated_hours if task.estimated_hours > 0 else 0
        )
        # High variance curve
        reward += (urgency_multiplier ** 1.5) * (fraction_done_this_step ** 1.2) * 5.0

        # ── 2. Low-priority-task penalty ────────────────────────────────────
        open_tasks = [t for t in new_state.tasks if t.status != "done"]
        urgent_open = [
            t
            for t in open_tasks
            if (t.type == "bug" and t.severity == "high") or t.deadline <= 2
        ]
        is_working_on_urgent = (
            task.type == "bug" and task.severity == "high"
        ) or task.deadline <= 2

        if urgent_open and not is_working_on_urgent:
            # Massive variance penalty based on how many urgent tasks are ignored
            reward -= 5.0 * len(urgent_open)

        # ── 3. Task-completion bonus ────────────────────────────────────────
        if task.status == "done":
            # Very high completion spike
            completion_bonus = 5.0 + (urgency_multiplier ** 2.0)
            reward += completion_bonus

        # ── 4. Ask-for-help bonus ───────────────────────────────────────────
        if action.ask_for_help:
            task_is_hard = (
                task.estimated_hours >= 5.0
                or (task.type == "bug" and task.severity == "high")
            )
            if task_is_hard:
                reward += 2.0   # strong positive signal
            else:
                reward -= 5.0   # harsh penalty for crying wolf

        # ── 5. Hour-spreading / thin-work penalty ───────────────────────────
        if (
            clamped_hours < 0.3 * task.estimated_hours
            and task.status != "done"
        ):
            reward -= 3.0  # severely punish fragmented work that doesn't finish

        # ── 6. Terminal rewards ─────────────────────────────────────────────
        if all_done:
            # Quadratic bonus for early finishes
            days_remaining = max(0, self.max_days - new_state.day)
            reward += 20.0 + (days_remaining ** 2.0) * 5.0

        if deadline_expired and not all_done:
            missed = sum(1 for t in new_state.tasks if t.status != "done")
            # Quadratic disaster penalty
            reward -= 20.0 + (missed ** 2.0) * 10.0

        return round(reward, 4)
