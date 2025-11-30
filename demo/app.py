"""
TeachRL Demo - Interactive Web Interface
Flask-based demo for visualizing the meta-curriculum learning system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, request, session, send_file, abort
from flask_cors import CORS
import os
import numpy as np
import json
import threading
import time
import random
import subprocess
from typing import Any, Dict

from tasks.task_generator import (
    generate_task, generate_task_by_arm, generate_eval_dataset,
    FAMILY_NAMES, DIFFICULTY_NAMES, arm_to_name, NUM_FAMILIES, NUM_DIFFICULTIES
)
from learning_profile import LearningProfile
from training.training_loop import (
    MetaTrainer,
    MetaTrainingConfig,
    run_random_curriculum,
    run_fixed_curriculum,
)
from student.ppo_agent import StudentAgent
from teacher.teacher_bandit import BanditStrategy

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "mentorflow-dev-secret")
# Allow frontend to send cookies for session-based difficulty control
CORS(app, supports_credentials=True)

# Mapping from UI subjects to generator family ids
SUBJECT_FAMILY_MAP = {
    "programming": list(range(0, 14)),  # All programming-related families
    "maths": [14],
    "physics": [15],
    "chemistry": [16],
}

learning_profile = LearningProfile(FAMILY_NAMES, DIFFICULTY_NAMES)

# Global state for training
training_state = {
    "running": False,
    "step": 0,
    "total_steps": 0,
    "accuracies": [],
    "rewards": [],
    "selected_arms": [],
    "arm_names": [],
    "curriculum_heatmap": np.zeros((NUM_FAMILIES, NUM_DIFFICULTIES)).tolist(),
    "selection_timeline": [],
    "final_accuracy": 0,
    "current_task": None,
    "baseline_random": None,
    "baseline_fixed": None,
}

CACHE_DIR = (Path(__file__).parent / "cache").resolve()
CACHE_PATH = CACHE_DIR / "last_run.json"


def _get_difficulty_state() -> Dict[str, Any]:
    state = session.get("difficulty_state") or {}
    history = state.get("history", [])
    state.setdefault("current_level", 1)
    state["history"] = history[-20:]
    if history:
        rolling = sum(history) / len(history)
    else:
        rolling = 0.0
    state["rolling_accuracy"] = round(rolling, 3)
    session["difficulty_state"] = state
    session.modified = True
    return state


def _update_difficulty_state(correct: bool) -> Dict[str, Any]:
    state = _get_difficulty_state()
    history = state.get("history", [])
    history.append(bool(correct))
    history = history[-20:]
    rolling = sum(history) / len(history)

    level = state.get("current_level", 1)
    if rolling > 0.8 and level < NUM_DIFFICULTIES:
        level += 1
    elif rolling < 0.4 and level > 1:
        level -= 1

    state.update(
        {
            "history": history,
            "rolling_accuracy": round(rolling, 3),
            "current_level": level,
        }
    )
    session["difficulty_state"] = state
    session.modified = True
    return state


def _build_training_response(state: Dict[str, Any]) -> Dict[str, Any]:
    arm_labels = [arm_to_name(i) for i in range(NUM_FAMILIES * NUM_DIFFICULTIES)]

    teacher_payload = {
        "accuracies": state.get("accuracies", []),
        "rewards": state.get("rewards", []),
        "selected_arms": state.get("selected_arms", []),
        "arm_names": state.get("arm_names", []),
        "final_accuracy": state.get("final_accuracy", 0.0),
        "current_task": state.get("current_task"),
        "heatmap": {
            "num_arms": NUM_FAMILIES * NUM_DIFFICULTIES,
            "num_steps": state.get("total_steps", 0),
            "timeline": _to_list(state.get("selection_timeline", [])),
            "cumulative": _to_list(state.get("curriculum_heatmap", [])),
            "arm_labels": arm_labels,
        },
        "metadata": {
            "step": state.get("step", 0),
            "total_steps": state.get("total_steps", 0),
            "running": state.get("running", False),
        },
    }

    def build_baseline(key: str) -> Dict[str, Any]:
        data = state.get(key)
        if not data:
            return {}
        accuracies = data.get("accuracy") or data.get("accuracies") or []
        return {
            "accuracies": accuracies,
            "rewards": [],
            "selected_arms": data.get("arms", []),
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "heatmap": {
                "num_arms": NUM_FAMILIES * NUM_DIFFICULTIES,
                "num_steps": data.get("meta_steps", 0),
                "timeline": _to_list(data.get("heatmap", [])),
                "arm_labels": arm_labels,
            },
            "metadata": {
                "step": len(accuracies),
                "total_steps": data.get("meta_steps", 0),
                "running": False,
            },
        }

    return {
        "teacher": teacher_payload,
        "baseline_random": build_baseline("baseline_random"),
        "baseline_fixed": build_baseline("baseline_fixed"),
    }


def _to_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _serialize_task(task, arm_id: int, *, subject_label: str | None = None) -> Dict[str, Any]:
    family_id = arm_id // NUM_DIFFICULTIES
    difficulty_id = task.difficulty_id if hasattr(task, "difficulty_id") else (arm_id % NUM_DIFFICULTIES) + 1
    difficulty_idx = max(0, int(difficulty_id) - 1)
    explanation = (
        f"The correct answer is {task.human_choices[task.correct_action]} "
        f"because it matches the adaptation the teacher expected at difficulty {DIFFICULTY_NAMES[difficulty_idx]}."
    )
    return {
        "subject": subject_label or FAMILY_NAMES[family_id],
        "family": FAMILY_NAMES[family_id],
        "difficulty": DIFFICULTY_NAMES[difficulty_idx],
        "difficulty_level": int(difficulty_id),
        "arm_id": arm_id,
        "prompt": task.human_prompt,
        "choices": task.human_choices,
        "answer_index": int(task.correct_action),
        "explanation": explanation,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/task/random')
def get_random_task():
    """Get a random task for demo."""
    family_id = np.random.randint(0, NUM_FAMILIES)
    difficulty_idx = np.random.randint(0, NUM_DIFFICULTIES)
    seed = np.random.randint(0, 10000)
    
    task = generate_task(family_id, difficulty_idx + 1, seed)
    
    return jsonify({
        "arm_id": family_id * NUM_DIFFICULTIES + difficulty_idx,
        "family": FAMILY_NAMES[family_id],
        "difficulty": DIFFICULTY_NAMES[difficulty_idx],
        "difficulty_level": int(difficulty_idx + 1),
        "prompt": task.human_prompt,
        "choices": task.human_choices,
        "correct_action": task.correct_action,
        "correct_answer": task.human_choices[task.correct_action],
    })


@app.route('/api/tasks/random')
def get_tasks_random():
    arm_id = np.random.randint(0, NUM_FAMILIES * NUM_DIFFICULTIES)
    task = generate_task_by_arm(int(arm_id), seed=np.random.randint(0, 10000))
    return jsonify(_serialize_task(task, int(arm_id)))


@app.route('/api/tasks/by-subject')
def get_tasks_by_subject():
    subject = (request.args.get('subject') or '').lower()
    families = SUBJECT_FAMILY_MAP.get(subject)
    if not families:
        return jsonify({"error": "Unknown subject"}), 404

    difficulty_name = (request.args.get('difficulty') or '').lower()
    if difficulty_name in DIFFICULTY_NAMES:
        difficulty_idx = DIFFICULTY_NAMES.index(difficulty_name)
    else:
        difficulty_idx = np.random.randint(0, NUM_DIFFICULTIES)

    family_id = int(np.random.choice(families))
    arm_id = family_id * NUM_DIFFICULTIES + difficulty_idx
    task = generate_task_by_arm(arm_id, seed=np.random.randint(0, 10000))
    return jsonify(_serialize_task(task, arm_id, subject_label=subject))


@app.route('/api/task/<int:arm_id>')
def get_task_by_arm(arm_id):
    """Get a task by arm ID."""
    seed = np.random.randint(0, 10000)
    task = generate_task_by_arm(arm_id, seed)
    family_id = arm_id // NUM_DIFFICULTIES
    difficulty_idx = arm_id % NUM_DIFFICULTIES
    
    return jsonify({
        "arm_id": arm_id,
        "family": FAMILY_NAMES[family_id],
        "difficulty": DIFFICULTY_NAMES[difficulty_idx],
        "difficulty_level": int(difficulty_idx + 1),
        "prompt": task.human_prompt,
        "choices": task.human_choices,
        "correct_action": task.correct_action,
        "correct_answer": task.human_choices[task.correct_action],
    })


@app.route('/api/task/check', methods=['POST'])
def check_answer():
    """Check if answer is correct."""
    data = request.json
    correct = data.get('selected') == data.get('correct_action')
    family = data.get('family')
    difficulty = data.get('difficulty')
    arm_id = data.get('arm_id')
    if family and difficulty:
        learning_profile.update_profile(
            family=family,
            difficulty=difficulty,
            arm_id=arm_id,
            correct=correct,
        )
    return jsonify({"correct": correct})


@app.route('/api/task/next')
def get_next_task():
    """Serve the next task using the adaptive difficulty controller."""
    state = _get_difficulty_state()
    level = state.get("current_level", 1)
    subject = (request.args.get("subject") or "").lower()
    family_param = request.args.get("family_id")

    if family_param is not None:
        try:
            family_id = int(family_param)
        except ValueError:
            return jsonify({"error": "Invalid family_id"}), 400
    else:
        families = SUBJECT_FAMILY_MAP.get(subject) or list(range(NUM_FAMILIES))
        family_id = int(np.random.choice(families))

    seed = np.random.randint(0, 10000)
    task = generate_task(family_id, level, seed)
    arm_id = family_id * NUM_DIFFICULTIES + (level - 1)
    return jsonify({
        "arm_id": int(arm_id),
        "family_id": family_id,
        "family": FAMILY_NAMES[family_id],
        "difficulty_level": level,
        "difficulty_label": DIFFICULTY_NAMES[level - 1],
        "rolling_accuracy": state.get("rolling_accuracy", 0.0),
        "history": state.get("history", []),
        "prompt": task.human_prompt,
        "choices": task.human_choices,
        "answer_index": int(task.correct_action),
        "topic": FAMILY_NAMES[family_id],
    })


@app.route('/api/user/attempt', methods=['POST'])
def record_user_attempt():
    """Record a user attempt and update difficulty."""
    data = request.json or {}
    correct = bool(data.get("correct"))
    prev_state = _get_difficulty_state()
    updated = _update_difficulty_state(correct)
    return jsonify({
        "status": "ok",
        "previous_level": prev_state.get("current_level", 1),
        "difficulty_state": updated,
        "level_changed": prev_state.get("current_level", 1) != updated.get("current_level", 1),
    })


@app.route('/api/profile')
def get_profile():
    return jsonify(learning_profile.serialize())


@app.route('/api/train/start', methods=['POST'])
def start_training():
    """Start meta-training."""
    global training_state
    
    if training_state["running"]:
        return jsonify({"error": "Training already running"}), 400
    
    data = request.json or {}
    num_steps = data.get('num_steps', 50)
    strategy = data.get('strategy', 'ucb1')
    replay_mode = bool(data.get('replay_mode'))

    # Replay from cache if available
    if replay_mode:
        if CACHE_PATH.exists():
            with open(CACHE_PATH, 'r') as f:
                cached = json.load(f)
            return jsonify(cached)
        return jsonify({"error": "No cached run available."}), 404
    
    # Map strategy string to enum
    strategy_map = {
        'ucb1': BanditStrategy.UCB1,
        'thompson': BanditStrategy.THOMPSON_SAMPLING,
        'epsilon_greedy': BanditStrategy.EPSILON_GREEDY,
        'random': BanditStrategy.RANDOM,
    }
    
    # Reset state
    num_arms = NUM_FAMILIES * NUM_DIFFICULTIES  # 15 arms
    training_state = {
        "running": True,
        "step": 0,
        "total_steps": num_steps,
        "accuracies": [],
        "rewards": [],
        "selected_arms": [],
        "arm_names": [],
        "curriculum_heatmap": np.zeros((NUM_FAMILIES, NUM_DIFFICULTIES)).tolist(),
        "selection_timeline": np.zeros((num_arms, num_steps), dtype=int).tolist(),  # [arm_id][step] = 1 if selected
        "final_accuracy": 0,
        "current_task": None,
        "baseline_random": None,
        "baseline_fixed": None,
    }
    
    # Start training in background thread
    def train_background():
        global training_state
        try:
            config = MetaTrainingConfig(
                num_meta_steps=num_steps,
                student_train_steps=128,
                eval_tasks_per_type=3,
                teacher_strategy=strategy_map.get(strategy, BanditStrategy.UCB1),
                seed=42,
                verbose=False,
            )
            
            trainer = MetaTrainer(config)
            
            # Custom training loop to update state
            for step in range(num_steps):
                if not training_state["running"]:
                    break
                
                # Evaluate before
                acc_before = trainer.student.evaluate(trainer.eval_tasks)
                
                # Teacher selects arm
                arm_id = trainer.teacher.select_arm()
                arm_name = arm_to_name(arm_id)
                
                # Get sample task for display
                task = generate_task_by_arm(arm_id, seed=step)
                training_state["current_task"] = {
                    "arm_name": arm_name,
                    "prompt": task.human_prompt,
                    "choices": task.human_choices,
                }
                
                # Train student
                trainer.student.train_on_task(arm_id, total_timesteps=config.student_train_steps)
                
                # Evaluate after
                acc_after = trainer.student.evaluate(trainer.eval_tasks)
                reward = acc_after - acc_before
                
                # Update teacher
                trainer.teacher.update(arm_id, reward)
                
                # Update state
                training_state["step"] = step + 1
                training_state["accuracies"].append(float(acc_after))
                training_state["rewards"].append(float(reward))
                training_state["selected_arms"].append(int(arm_id))
                training_state["arm_names"].append(arm_name)
                
                # Update heatmap (cumulative counts)
                family_id = arm_id // NUM_DIFFICULTIES
                difficulty_id = arm_id % NUM_DIFFICULTIES
                heatmap = np.array(training_state["curriculum_heatmap"])
                heatmap[family_id, difficulty_id] += 1
                training_state["curriculum_heatmap"] = heatmap.tolist()
                
                # Update selection timeline: [arm_id][step] = 1
                timeline = training_state["selection_timeline"]
                timeline[arm_id][step] = 1
                
                training_state["final_accuracy"] = float(acc_after)
                
                print(f"Step {step+1}/{num_steps} | Acc: {acc_after:.2%} | Arm: {arm_name}")
            
            # Teacher training complete
            print("Teacher training complete!")
            
            # Run random curriculum baseline using helper
            baseline_agent = StudentAgent(
                learning_rate=config.learning_rate,
                n_steps=config.ppo_n_steps,
                batch_size=config.ppo_batch_size,
                seed=config.seed + 101,
                verbose=0
            )
            baseline_rng = random.Random(config.seed + 999)
            baseline_result = run_random_curriculum(
                meta_steps=num_steps,
                model=baseline_agent,
                env={"train_steps": config.student_train_steps},
                eval_set=trainer.eval_tasks,
                num_arms=num_arms,
                rng=baseline_rng,
            )
            training_state["baseline_random"] = baseline_result
            
            # Run fixed curriculum baseline using helper
            fixed_agent = StudentAgent(
                learning_rate=config.learning_rate,
                n_steps=config.ppo_n_steps,
                batch_size=config.ppo_batch_size,
                seed=config.seed + 202,
                verbose=0
            )
            fixed_result = run_fixed_curriculum(
                meta_steps=num_steps,
                model=fixed_agent,
                env={"train_steps": config.student_train_steps},
                eval_set=trainer.eval_tasks,
                num_arms=num_arms,
            )
            training_state["baseline_fixed"] = fixed_result
            
            training_state["running"] = False
            print("Baseline complete!")

            # Persist snapshot for replay mode
            snapshot = _build_training_response(training_state)
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(CACHE_PATH, 'w') as f:
                json.dump(snapshot, f)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            training_state["running"] = False
            training_state["error"] = str(e)
            print(f"Training error: {e}")
    
    # Start teacher thread
    teacher_thread = threading.Thread(target=train_background)
    teacher_thread.start()
    
    return jsonify({"status": "started", "num_steps": num_steps})


@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    """Stop training."""
    global training_state
    training_state["running"] = False
    return jsonify({"status": "stopped"})


@app.route('/api/train/status')
def training_status():
    """Get structured results for teacher + baselines."""
    return jsonify(_build_training_response(training_state))


@app.route('/api/train/progress')
def training_progress():
    """Return basic progress metadata."""
    metadata = training_state.get("step", 0)
    total = training_state.get("total_steps", 0) or 1
    return jsonify({
        "step": metadata,
        "total_steps": total,
        "running": training_state.get("running", False),
    })


@app.route('/api/train/step', methods=['POST'])
def training_step_route():
    """Trigger a lightweight heartbeat for training clients."""
    return jsonify(training_progress().json)


@app.route('/api/families')
def get_families():
    """Get task family info."""
    families = []
    for i, name in enumerate(FAMILY_NAMES):
        families.append({
            "id": i,
            "name": name,
            "difficulties": [
                {"id": j, "name": DIFFICULTY_NAMES[j], "arm_id": i * NUM_DIFFICULTIES + j}
                for j in range(NUM_DIFFICULTIES)
            ]
        })
    return jsonify(families)


@app.route('/api/compare/run', methods=['POST'])
def run_strategy_comparison():
    """
    Run compare_strategies and expose the plot for the frontend.
    Uses mock student + medium generator for speed/stability.
    """
    cmd = [
        sys.executable,
        "teacher_agent_dev/compare_strategies.py",
        "--iterations", os.environ.get("COMPARE_ITERATIONS", "100"),
        "--use-mock-student",
        "--medium-generator",
        "--resample-eval",
        "--deterministic",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            return jsonify({
                "error": "compare_strategies failed",
                "stderr": result.stderr[-500:]
            }), 500
        return jsonify({
            "status": "ok",
            "plot_url": "/api/compare/plot",
            "stdout": result.stdout[-1000:],  # tail for debugging
        })
    except subprocess.TimeoutExpired:
        return jsonify({"error": "compare_strategies timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/compare/plot')
def get_compare_plot():
    """Serve the latest comparison plot if available."""
    plot_path = Path(__file__).parent.parent / "comparison_all_strategies.png"
    if not plot_path.exists():
        abort(404)
    return send_file(plot_path, mimetype="image/png")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
