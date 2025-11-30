# MentorFlow: RL Teacher-Student System (Demo)

## What This Project Delivers
- **RL Teacher with Curriculum Discovery**: UCB-based teacher learns optimal task ordering across topics/difficulties; bandit-driven exploration vs. exploitation, exposed via `/api/train/*` for live dashboards.
- **Configurable Students**:
  - **Mock Student** (fast): Ebbinghaus-style forgetting, per-topic skill, transfer learning; ideal for demos and plots.
  - **PPO Student** (RL): Stable Baselines PPO wrapper over `StudentEnv` with improved training signal, stochasticity, and forgetting; toggleable in the UI/CLI.
- **Task Generators**:
  - **Medium Generator (default)**: STEM-heavy, LM-light, multiple templates per topic/difficulty for a large bank without language-model reliance.
  - **Legacy Mock Generator**: Simple baseline, still available.
- **Strategy Comparisons**: Random vs. Progressive vs. Teacher curriculum, with resampled evals for variance, difficulty-aware plots, and PNG export.
- **Frontend Integration**: Next.js frontend proxies to Flask backend (`demo/app.py`) for live teacher stats, training controls, and status dashboards.

## How to Run (Local Demo)
1) **Backend (Flask)**
```bash
cd /Users/leonardowang/MentorFlow
FLASK_APP=demo/app.py FLASK_ENV=production python3 demo/app.py  # listens on 5050
```

2) **Frontend (Next.js)**
```bash
cd frontend
FLASK_API_BASE_URL=http://localhost:5050 npm install   # first time
FLASK_API_BASE_URL=http://localhost:5050 npm run dev    # or npm run build && npm run start
```
Open the shown URL; training dashboards will hit `/api/train/*` through the proxy.

## How to Run Strategy Comparisons (CLI)
- Fast mock student + medium generator (default):
```bash
./venv/bin/python teacher_agent_dev/compare_strategies.py --iterations 200 --deterministic
```
- Toggle PPO student:
```bash
./venv/bin/python teacher_agent_dev/compare_strategies.py --iterations 200 --deterministic --resample-eval
```
- Legacy mock generator:
```bash
./venv/bin/python teacher_agent_dev/compare_strategies.py --iterations 200 --use-mock-generator --use-mock-student
```
Flags: `--use-mock-student`, `--use-mock-generator`, `--resample-eval`, `--seed`, `--iterations`.

## Notable Engineering Points
- **Medium Task Generator**: Multiple templates per topic/difficulty; LM-light; supports variance without heavy models.
- **PPO Wrapper Upgrades**: Longer rollouts, higher entropy for exploration, more eval episodes, stochastic predictions, and explicit forgetting to avoid linear/flat curves.
- **Plot Hardening**: Smoothing guards for short runs; resampled eval sets to inject variance.
- **Deployment Safety**: HF deploy scripts gated; primary flow is local/backend + Next frontend.
- **Frontend Proxy**: Next API routes point to `FLASK_API_BASE_URL` (default `http://localhost:5050`).

## Tests / Smoke
- Quick teacher tests: `./venv/bin/python teacher_agent_dev/test_teacher.py`
- End-to-end comparison: see CLI commands above (produces `comparison_all_strategies.png`).

## What to Demo
- Launch backend + frontend; start training via UI to show live teacher metrics (accuracies, rewards, curriculum heatmap).
- Run strategy comparison and display the saved plot.
- Adjust retention/strategy toggles in the UI to show student/teacher adaptability.
