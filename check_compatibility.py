#!/usr/bin/env python3
"""
Lightweight Compatibility Check - No Heavy Model Loading
Checks imports, interfaces, and basic functionality without loading large models
"""

import sys
import importlib
from pathlib import Path

def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"  ✅ {description}")
        return True
    except ImportError as e:
        print(f"  ❌ {description}: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  {description}: {e}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    path = Path(filepath)
    if path.exists():
        print(f"  ✅ {description}")
        return True
    else:
        print(f"  ❌ {description}: File not found")
        return False

def main():
    print("="*70)
    print("COMPATIBILITY CHECK - LIGHTWEIGHT")
    print("="*70)
    print()
    
    issues = []
    
    # Check file structure
    print("FILE STRUCTURE:")
    print("-" * 70)
    files_to_check = [
        ("tasks/task_generator.py", "Task generator"),
        ("student/student_env.py", "Student environment"),
        ("student/ppo_agent.py", "PPO agent"),
        ("teacher/teacher_bandit.py", "Teacher bandit"),
        ("training/training_loop.py", "Training loop"),
        ("teacher_agent_dev/interfaces.py", "Teacher agent interfaces"),
        ("student_agent_dev/interfaces.py", "Student agent interfaces"),
    ]
    
    for filepath, desc in files_to_check:
        if not check_file_exists(filepath, desc):
            issues.append(f"Missing file: {filepath}")
    
    # Check imports (lightweight)
    print("\nIMPORT CHECKS:")
    print("-" * 70)
    
    # Core imports
    if not check_import("tasks.task_generator", "Task generator module"):
        issues.append("Cannot import task_generator")
    
    if not check_import("student.student_env", "Student environment module"):
        issues.append("Cannot import student_env")
    
    # Check interface compatibility
    print("\nINTERFACE COMPATIBILITY:")
    print("-" * 70)
    
    try:
        sys.path.insert(0, str(Path("teacher_agent_dev").absolute()))
        sys.path.insert(0, str(Path("student_agent_dev").absolute()))
        
        from teacher_agent_dev.interfaces import Task as TTask, StudentState as TState
        from student_agent_dev.interfaces import Task as STask, StudentState as SState
        
        import inspect
        
        # Check Task fields
        t_task_fields = set(inspect.signature(TTask.__init__).parameters.keys()) - {'self'}
        s_task_fields = set(inspect.signature(STask.__init__).parameters.keys()) - {'self'}
        
        if t_task_fields == s_task_fields:
            print("  ✅ Task dataclass fields match")
        else:
            diff = t_task_fields.symmetric_difference(s_task_fields)
            print(f"  ⚠️  Task fields differ: {diff}")
            issues.append(f"Task interface mismatch: {diff}")
        
        # Check StudentState fields
        t_state_fields = set(inspect.signature(TState.__init__).parameters.keys()) - {'self'}
        s_state_fields = set(inspect.signature(SState.__init__).parameters.keys()) - {'self'}
        
        if t_state_fields == s_state_fields:
            print("  ✅ StudentState dataclass fields match")
        else:
            diff = t_state_fields.symmetric_difference(s_state_fields)
            print(f"  ⚠️  StudentState fields differ: {diff}")
            issues.append(f"StudentState interface mismatch: {diff}")
            
    except Exception as e:
        print(f"  ❌ Interface compatibility check failed: {e}")
        issues.append(f"Interface check error: {e}")
    
    # Check basic task generation (no model loading)
    print("\nBASIC FUNCTIONALITY:")
    print("-" * 70)
    
    try:
        from tasks.task_generator import generate_task, NUM_FAMILIES, NUM_DIFFICULTIES
        task = generate_task(family_id=0, difficulty_id=0, seed=42)
        if task and hasattr(task, 'human_prompt'):
            print(f"  ✅ Task generation works ({NUM_FAMILIES} families × {NUM_DIFFICULTIES} difficulties)")
        else:
            print("  ❌ Task generation returned invalid task")
            issues.append("Task generation issue")
    except Exception as e:
        print(f"  ❌ Task generation failed: {e}")
        issues.append(f"Task generation error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if issues:
        print(f"⚠️  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All compatibility checks passed!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

