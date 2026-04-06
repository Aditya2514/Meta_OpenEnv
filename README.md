---
title: Assignment Planner Environment
emoji: 🏢
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Assignment & Bug-Fix Planner Agent for Students and Junior Developers

> An **OpenEnv-compliant AI-agent task** in which a student or junior developer
> must juggle coding assignments and production bugs under tight deadlines —
> without burning out.

---

## Problem Statement

A student or junior developer has **multiple coding tasks and bugs** under a **deadline**
and **limited daily working hours**. An AI agent must decide:

- **which task** to work on,
- **how many hours** to spend, and
- **whether to ask for help**,

while avoiding **burnout** and **missed deadlines**.

The environment exposes a gym-style interface (`reset → step* → done`) and is served
as an **OpenEnv-compliant HTTP API** deployable on Hugging Face Spaces.

---

## Repository Structure

```
assignment-planner-env/
├── inference.py                          ← Baseline inference script (Phase 4)
├── requirements.txt                      ← Python dependencies
├── README.md
└── src/envs/assignment_planner/
    ├── models.py                         ← Pydantic models (Task, Action, …)
    ├── task_config.py                    ← Fixed episode configurations
    ├── environment.py                    ← AssignmentPlannerEnv (core logic)
    ├── graders.py                        ← Deterministic 0–1 graders
    ├── smoke_test.py                     ← Quick sanity check
    └── server/
        ├── app.py                        ← FastAPI HTTP server
        ├── openenv.yaml                  ← OpenEnv metadata spec
        └── Dockerfile                    ← HF Spaces Docker image
```

---

## Action & Observation Spaces

### Action

| Field | Type | Constraints | Description |
|---|---|---|---|
| `task_id` | `int` | `0 ≤ task_id < len(tasks)`, task not `"done"` | Index of the task to work on |
| `hours` | `float` | `0 < hours ≤ min(hours_left_today, remaining_hours[task_id])` | Hours to invest this step |
| `ask_for_help` | `bool` | — | Request senior/mentor assistance |

### Observation

| Field | Type | Description |
|---|---|---|
| `day` | `int` | Current day of the episode (0-indexed) |
| `hours_left_today` | `float` | Remaining work hours available today |
| `tasks` | `List[Task]` | Full snapshot of all tasks |
| `summary.tasks_remaining` | `int` | Number of unfinished tasks |
| `summary.high_severity_bugs_remaining` | `int` | Open critical bugs |
| `summary.days_until_deadline` | `int` | Minimum days-to-deadline across open tasks |

### Task Fields

| Field | Type | Notes |
|---|---|---|
| `id` | `int` | Unique index |
| `name` | `str` | Human-readable label |
| `type` | `"bug"` \| `"feature"` \| `"review"` | Task category |
| `severity` | `"high"` \| `"medium"` \| `"low"` \| `None` | Only for bugs |
| `deadline` | `int` | Days remaining until due |
| `estimated_hours` | `float` | Original effort estimate |
| `remaining_hours` | `float` | Hours left to complete |
| `status` | `"not_started"` \| `"in_progress"` \| `"done"` | Progress state |

---

## Task Descriptions

### `easy_1` — Basic Task Selection
**2 tasks · 3 days · 6 h/day · Total work: 6 h**

| # | Task | Type | Severity | Deadline | Est. Hours |
|---|---|---|---|---|---|
| 0 | Fix high-severity login bug | `bug` | `high` | 3 days | 2 h |
| 1 | Implement basic dashboard feature | `feature` | — | 3 days | 4 h |

The agent has 18 h of total capacity for 6 h of work. Challenge: learn to prioritise the
high-severity bug over the feature.

---

### `medium_1` — Time-Aware Prioritisation
**3 tasks · 4 days · 6 h/day · Total work: 11 h**

| # | Task | Type | Severity | Deadline | Est. Hours |
|---|---|---|---|---|---|
| 0 | Fix high-severity API crash on /checkout | `bug` | `high` | **2 days** | 3 h |
| 1 | Implement profile settings page | `feature` | — | 4 days | 3 h |
| 2 | Refactor auth module (low-priority) | `feature` | — | 4 days | 5 h |

The critical bug **must** be finished within day 2 or it fails the grader. The agent must
balance urgency with feature completion.

---

### `hard_1` — Multi-Day Triage Under Capacity Pressure
**5 tasks · 3 days · 6 h/day · Total work: 21 h (capacity: 18 h)**

| # | Task | Type | Severity | Deadline | Est. Hours |
|---|---|---|---|---|---|
| 0 | CRITICAL: memory leak in data pipeline | `bug` | `high` | **1 day** | 4 h |
| 1 | CRITICAL: race condition in websocket handler | `bug` | `high` | **2 days** | 4 h |
| 2 | Fix: incorrect pagination offset | `bug` | `medium` | 3 days | 2 h |
| 3 | Migrate legacy endpoints to REST v2 | `feature` | — | 3 days | 7 h |
| 4 | Review security audit report | `review` | — | 3 days | 4 h |

Total work *exceeds* available capacity by 3 h. The agent must triage: both critical bugs
must be resolved on time while accepting that some lower-priority tasks will be missed.

---

## Reward & Grader Design

### Step Reward (Dense, Shaped)

Every `step()` returns a continuous reward shaped to guide the agent:

| Component | Effect |
|---|---|
| **Urgency bonus** | `+` proportional to fraction completed × urgency multiplier (high-bug = ×2.5, near-deadline = ×2.0) |
| **Low-priority penalty** | `−0.5` if working on non-urgent task while urgent tasks remain open |
| **Completion bonus** | `+1.0 + urgency × 0.5` when a task reaches `status = "done"` |
| **Ask-for-help bonus** | `+0.2` when asking help on genuinely hard tasks (≥5 h or high-severity bug) |
| **Thin-work penalty** | `−0.2` if hours spent < 30 % of task estimate without finishing |
| **Terminal bonus** | `+5.0 + 0.5 × days_remaining` when all tasks finish before deadline |
| **Deadline penalty** | `−3.0 − missed_tasks` when the episode ends with open tasks |

### Episode Graders (0.0 – 1.0)

Three deterministic graders produce the final judge score from the episode trajectory:

```
final_score = clip(
    α × score_bugs + β × score_features
    − γ × workload_ratio − δ × bug_ignored
    + bonus,
    0.0, 1.0
)
```

| Metric | Definition |
|---|---|
| `score_bugs` | Fraction of high-severity bugs finished **on time** |
| `score_features` | Fraction of features finished (any time before episode end) |
| `workload_ratio` | Fraction of days where hours worked > 8 h |
| `bug_ignored` | `1` if any high-severity bug was completely untouched until the final day |
| `burnout_free` | All days ≤ 8 h worked |
| `balanced_work` | Both bugs and features were touched during the episode |

#### Grader Coefficients

| Grader | α | β | γ | δ | Bonus |
|---|---|---|---|---|---|
| `grade_easy` | 0.70 | 0.30 | 0.05 | 0.10 | none |
| `grade_medium` | 0.55 | 0.25 | 0.20 | 0.20 | +0.10 for balanced work |
| `grade_hard` | 0.60 | 0.15 | 0.30 | 0.25 | +0.15 for burnout-free |

---

## Setup & Usage

### Prerequisites

```bash
pip install pydantic fastapi "uvicorn[standard]" openai pyyaml
```

### Option 1 — Local Python (no Docker)

```bash
git clone https://github.com/YOUR_USERNAME/assignment-planner-env
cd assignment-planner-env

pip install -r requirements.txt

# Run smoke test (heuristic agent, all 3 episodes)
python src/envs/assignment_planner/smoke_test.py

# Run baseline inference (heuristic, no LLM)
python inference.py --local --no-llm
```

### Option 2 — Docker (local or HF Spaces)

```bash
# Build from repo root
docker build -t assignment-planner \
  -f src/envs/assignment_planner/server/Dockerfile .

# Run locally
docker run -p 7860:7860 \
  -e DEFAULT_TASK=easy_1 \
  assignment-planner

# Verify health
curl http://localhost:7860/
# → {"status":"ok","current_task_id":"easy_1","available_tasks":["easy_1","medium_1","hard_1"]}

# Interactive API docs
open http://localhost:7860/docs
```

### Option 3 — HF Spaces Deployment

1. Create a new HF Space with **Docker** backend.
2. Push this repository as the Space source.
3. Add tags: `openenv`, `agent-environment`, `task-planning`.
4. HF Spaces will build the Dockerfile and expose port 7860.

### Running Inference Against the Server

```bash
# Against a live server (HF Space or local Docker)
export API_BASE_URL="https://YOUR-SPACE.hf.space"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
python inference.py

# Against local Docker with heuristic fallback (no LLM token needed)
API_BASE_URL="http://localhost:7860" python inference.py --no-llm

# Fully local (no server, no LLM) — fastest sanity check
python inference.py --local --no-llm

# Specific tasks only
python inference.py --local --no-llm --tasks easy_1 hard_1
```

---

## HTTP API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check + available task IDs |
| `/reset?task_id=<id>` | POST | Initialise episode, returns `Observation` |
| `/step` | POST | Apply `Action`, returns `{observation, reward, done, info}` |
| `/state` | GET | Full internal `State` (for logging/debugging) |
| `/docs` | GET | Swagger UI with JSON schemas |
| `/redoc` | GET | ReDoc UI |

**Example `/step` request body:**
```json
{
  "task_id": 0,
  "hours": 3.0,
  "ask_for_help": true
}
```

---

## Example Run Output

```text
2026-04-02 23:05:59 [INFO] ============================================================
2026-04-02 23:05:59 [INFO] Assignment & Bug-Fix Planner Agent – Baseline Inference
2026-04-02 23:05:59 [INFO] ============================================================
2026-04-02 23:05:59 [INFO] Mode       : local env
2026-04-02 23:05:59 [INFO] LLM        : heuristic (no LLM)
2026-04-02 23:05:59 [INFO] Tasks      : ['easy_1', 'medium_1', 'hard_1']

[START] task=easy_1, env_url=local, model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1, action={"task_id":0,"hours":2.0,"ask_for_help":true}, reward=2.875, done=False, info={"clamped_hours":2.0,"day_advanced":false,"all_tasks_done":false,"deadline_expired":false}
[STEP] step=2, action={"task_id":1,"hours":4.0,"ask_for_help":false}, reward=11.15, done=True, info={"clamped_hours":4.0,"day_advanced":true,"all_tasks_done":true,"deadline_expired":false}
[END] score=1.0

[START] task=medium_1, env_url=local, model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1, action={"task_id":0,"hours":3.0,"ask_for_help":true}, reward=3.575, done=False, info={"clamped_hours":3.0,"day_advanced":false,...}
[STEP] step=2, action={"task_id":1,"hours":3.0,"ask_for_help":false}, reward=2.2, done=False, info={"clamped_hours":3.0,"day_advanced":true,...}
[STEP] step=3, action={"task_id":2,"hours":5.0,"ask_for_help":true}, reward=13.375, done=True, info={"clamped_hours":5.0,"day_advanced":true,"all_tasks_done":true,...}
[END] score=1.0

[START] task=hard_1, env_url=local, model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1, action={"task_id":0,"hours":4.0,"ask_for_help":true}, reward=6.45, done=False, info={"clamped_hours":4.0,...}
[STEP] step=2, action={"task_id":1,"hours":2.0,"ask_for_help":true}, reward=2.2875, done=False, info={"clamped_hours":2.0,...}
[STEP] step=3, action={"task_id":1,"hours":2.0,"ask_for_help":true}, reward=2.1375, done=False, info={"clamped_hours":2.0,...}
[STEP] step=4, action={"task_id":2,"hours":2.0,"ask_for_help":false}, reward=2.0625, done=False, info={"clamped_hours":2.0,...}
[STEP] step=5, action={"task_id":3,"hours":6.0,"ask_for_help":true}, reward=2.4038, done=False, info={"clamped_hours":6.0,...}
[STEP] step=6, action={"task_id":3,"hours":1.0,"ask_for_help":true}, reward=-3.4, done=True, info={"clamped_hours":1.0,"deadline_expired":true,...}
[END] score=0.75

==================================================
FINAL SCORES
==================================================
  easy_1        score=1.0000  |████████████████████|
  medium_1      score=1.0000  |████████████████████|
  hard_1        score=0.7500  |███████████████     |

  Mean score  : 0.9167
  Total time  : 0.8s
==================================================
```

---

## Baseline Scores

Scores from the **heuristic greedy agent** (no LLM, `--local --no-llm`):

| Task | Score | Notes |
|---|---|---|
| `easy_1` | **1.0000** | Both tasks finished on time; burnout-free |
| `medium_1` | **1.0000** | Critical bug done within 2-day window; balanced work bonus earned |
| `hard_1` | **0.7500** | Both critical bugs resolved; migration + review missed (total work > capacity by design) |
| **Mean** | **0.9167** | — |

> **Note:** An LLM agent is expected to score **lower** than the heuristic on hard_1 due
> to suboptimal hour allocation, but should score competitively on easy_1 and medium_1
> once the prompt includes clear prioritisation guidance.

---

## Architecture Overview

```
┌─────────────┐     POST /reset    ┌─────────────────────────┐
│  AI Agent   │ ─────────────────► │  FastAPI Server (app.py) │
│ (inference) │                    │  POST /step              │
│             │ ◄───────────────── │  GET  /state             │
└─────────────┘   Observation /    └──────────┬──────────────┘
                  reward / done               │ calls
                                   ┌──────────▼──────────────┐
                                   │  AssignmentPlannerEnv   │
                                   │  (environment.py)       │
                                   ├─────────────────────────┤
                                   │  task_config.py         │
                                   │  graders.py             │
                                   │  models.py              │
                                   └─────────────────────────┘
```

---

## License

MIT — see `LICENSE` for details.

---

## Citation

If you use this environment in research, please cite:

```bibtex
@misc{assignment-planner-env,
  title  = {Assignment \& Bug-Fix Planner Agent Environment},
  year   = {2026},
  note   = {OpenEnv-compliant task environment for AI agent evaluation}
}
```
