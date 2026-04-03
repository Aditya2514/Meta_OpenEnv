"""
models.py
---------
Pydantic models for the Assignment & Bug-Fix Planner Agent environment.

These models define the data shapes used for tasks, agent actions,
environment observations, and internal state throughout the episode.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Task(BaseModel):
    """
    Represents a single coding task or bug assigned to a student / junior dev.

    Attributes
    ----------
    id : int
        Unique identifier for the task.
    name : str
        Human-readable task label (e.g., "Fix null-pointer in auth module").
    type : Literal["feature", "bug", "review"]
        Category of the task.
    severity : Optional[Literal["low", "medium", "high"]]
        Relevant only for bugs; indicates urgency. None for features/reviews.
    deadline : int
        Number of days remaining until this task must be completed.
    estimated_hours : float
        Original estimate of total hours required to complete the task.
    remaining_hours : float
        Hours still needed to finish (decreases as work is done).
    status : Literal["not_started", "in_progress", "done"]
        Current progress state of the task.
    """

    id: int
    name: str
    type: Literal["feature", "bug", "review"]
    severity: Optional[Literal["low", "medium", "high"]] = None  # only for bugs
    deadline: int       # days remaining
    estimated_hours: float
    remaining_hours: float
    status: Literal["not_started", "in_progress", "done"]


class Action(BaseModel):
    """
    The action an agent takes during a single step.

    Attributes
    ----------
    task_id : int
        Index into the current task list (0-based) the agent chooses to work on.
    hours : float
        Hours to invest this step. Must satisfy 0 < hours <= min(hours_left_today,
        remaining_hours[task_id]).
    ask_for_help : bool
        Whether the agent decides to ask a senior / mentor for help on this task.
        This can yield a bonus reward on difficult tasks.
    """

    task_id: int
    hours: float = Field(..., gt=0, description="Hours to spend; must be > 0")
    ask_for_help: bool


class Summary(BaseModel):
    """
    High-level statistics summarising the current episode state.

    Attributes
    ----------
    tasks_remaining : int
        Number of tasks not yet marked "done".
    high_severity_bugs_remaining : int
        Count of unresolved bugs with severity == "high".
    days_until_deadline : int
        Minimum days_until_deadline across all remaining (non-done) tasks,
        or 0 if all tasks are done.
    """

    tasks_remaining: int
    high_severity_bugs_remaining: int
    days_until_deadline: int


class Observation(BaseModel):
    """
    What the agent perceives at every step (returned by reset() and step()).

    Attributes
    ----------
    day : int
        Current day of the episode (0-indexed).
    hours_left_today : float
        Remaining work hours available today.
    tasks : List[Task]
        Full snapshot of every task and its current state.
    summary : Summary
        Aggregated statistics for quick agent decision-making.
    """

    day: int
    hours_left_today: float
    tasks: List[Task]
    summary: Summary


class State(BaseModel):
    """
    Full internal environment state (used for debugging / serialisation).

    Attributes
    ----------
    day : int
        Current day of the episode.
    hours_left_today : float
        Remaining work capacity for the current day.
    tasks : List[Task]
        Complete task list with up-to-date remaining_hours and statuses.
    """

    day: int
    hours_left_today: float
    tasks: List[Task]
