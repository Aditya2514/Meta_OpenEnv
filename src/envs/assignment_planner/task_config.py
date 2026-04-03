"""
task_config.py
--------------
Canonical, fixed task-episode configurations for the Assignment & Bug-Fix
Planner Agent environment.

Each top-level key is a ``task_id`` string recognised by
``AssignmentPlannerEnv``.  Every config entry specifies:

  • ``max_days``       – maximum episode length in calendar days.
  • ``daily_capacity`` – available working hours per day.
  • ``tasks``          – ordered list of task dicts.

The ``tasks`` list uses the same field schema as the ``Task`` Pydantic model,
*excluding* the mutable fields ``remaining_hours`` and ``status`` (those are
always initialised by ``AssignmentPlannerEnv.reset()``).

Difficulty design rationale
----------------------------
easy_1  : 2 tasks, generous timeline (3 days, 6 h/day = 18 h total vs 6 h work).
medium_1: 3 tasks including one high-severity bug with a tighter deadline.
hard_1  : 5 tasks, two critical bugs due on day 1 & 2, total work > capacity.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Type alias for a single task definition (without run-time state fields)
# ---------------------------------------------------------------------------
TaskDef = Dict[str, Any]

# ---------------------------------------------------------------------------
# Master configuration
# ---------------------------------------------------------------------------
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ── easy_1 ───────────────────────────────────────────────────────────────
    # Two tasks: one high-severity bug + one small feature.
    # Total work = 6 h; capacity = 3 days × 6 h = 18 h  →  very achievable.
    "easy_1": {
        "max_days": 3,
        "daily_capacity": 6.0,
        "tasks": [
            {
                "id": 0,
                "name": "Fix high-severity login bug",
                "type": "bug",
                "severity": "high",
                "deadline": 3,          # 3 days to fix
                "estimated_hours": 2.0,
            },
            {
                "id": 1,
                "name": "Implement basic dashboard feature",
                "type": "feature",
                "severity": None,
                "deadline": 3,
                "estimated_hours": 4.0,
            },
        ],
    },

    # ── medium_1 ─────────────────────────────────────────────────────────────
    # Three tasks: critical API bug (tight deadline), a feature, a low-priority
    # refactor.  Total work = 11 h; capacity = 4 days × 6 h = 24 h.
    # Challenge: the high-severity bug must be finished within day 2.
    "medium_1": {
        "max_days": 4,
        "daily_capacity": 6.0,
        "tasks": [
            {
                "id": 0,
                "name": "Fix high-severity API crash on /checkout",
                "type": "bug",
                "severity": "high",
                "deadline": 2,          # must finish within 2 days
                "estimated_hours": 3.0,
            },
            {
                "id": 1,
                "name": "Implement profile settings page",
                "type": "feature",
                "severity": None,
                "deadline": 4,
                "estimated_hours": 3.0,
            },
            {
                "id": 2,
                "name": "Refactor auth module (low-priority cleanup)",
                "type": "feature",
                "severity": None,
                "deadline": 4,
                "estimated_hours": 5.0,
            },
        ],
    },

    # ── hard_1 ───────────────────────────────────────────────────────────────
    # Five tasks: two critical bugs with back-to-back deadlines, one medium bug,
    # one large feature, one review.
    # Total work = 21 h; capacity = 3 days × 6 h = 18 h  → must triage carefully.
    "hard_1": {
        "max_days": 3,
        "daily_capacity": 6.0,
        "tasks": [
            {
                "id": 0,
                "name": "CRITICAL: memory leak in data pipeline",
                "type": "bug",
                "severity": "high",
                "deadline": 1,          # must be done by end of day 1
                "estimated_hours": 4.0,
            },
            {
                "id": 1,
                "name": "CRITICAL: race condition in websocket handler",
                "type": "bug",
                "severity": "high",
                "deadline": 2,          # must be done by end of day 2
                "estimated_hours": 4.0,
            },
            {
                "id": 2,
                "name": "Fix medium: incorrect pagination offset",
                "type": "bug",
                "severity": "medium",
                "deadline": 3,
                "estimated_hours": 2.0,
            },
            {
                "id": 3,
                "name": "Migrate legacy endpoints to REST v2",
                "type": "feature",
                "severity": None,
                "deadline": 3,
                "estimated_hours": 7.0,
            },
            {
                "id": 4,
                "name": "Review security audit report",
                "type": "review",
                "severity": None,
                "deadline": 3,
                "estimated_hours": 4.0,
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def list_task_ids() -> List[str]:
    """Return all registered task_id strings."""
    return list(TASK_CONFIGS.keys())


def get_config(task_id: str) -> Dict[str, Any]:
    """
    Retrieve the full configuration for a given task_id.

    Raises
    ------
    KeyError
        If ``task_id`` is not found in TASK_CONFIGS.
    """
    if task_id not in TASK_CONFIGS:
        raise KeyError(
            f"Unknown task_id '{task_id}'. "
            f"Available: {list_task_ids()}"
        )
    return TASK_CONFIGS[task_id]
