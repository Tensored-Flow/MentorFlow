#!/usr/bin/env python3
"""
Safe Pipeline Test - Avoids problematic imports
Tests compatibility and functionality without triggering mutex errors
"""

import sys
import ast
from pathlib import Path

def check_file_structure():
    """Check if key files exist."""
    print("="*70)
    print("FILE STRUCTURE CHECK")
    print("="*70)
    
    key_files = {
        "tasks/task_generator.py": "Task generator",
        "student/student_env.py": "Student environment",
        "student/ppo_agent.py": "PPO agent",
        "student/slm_agent.py": "SLM agent",
        "teacher/teacher_bandit.py": "Teacher bandit",
        "training/training_loop.py": "Training loop",
        "teacher_agent_dev/interfaces.py": "Teacher interfaces",
        "student_agent_dev/interfaces.py": "Student interfaces",
    }
    
    all_exist = True
    for filepath, desc in key_files.items():
        path = Path(filepath)
        if path.exists():
            print(f"  ✅ {desc}: {filepath}")
        else:
            print(f"  ❌ {desc}: {filepath} - NOT FOUND")
            all_exist = False
    
    return all_exist

def check_task_generator_constants():
    """Check task generator constants by parsing file."""
    print("\n" + "="*70)
    print("TASK GENERATOR CONSTANTS")
    print("="*70)
    
    try:
        tg_file = Path("tasks/task_generator.py")
        content = tg_file.read_text()
        
        # Extract constants using AST
        tree = ast.parse(content)
        constants = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ['NUM_FAMILIES', 'NUM_DIFFICULTIES']:
                        if isinstance(node.value, ast.Constant):
                            constants[target.id] = node.value.value
        
        if 'NUM_FAMILIES' in constants and 'NUM_DIFFICULTIES' in constants:
            num_families = constants['NUM_FAMILIES']
            num_difficulties = constants['NUM_DIFFICULTIES']
            print(f"  ✅ NUM_FAMILIES: {num_families}")
            print(f"  ✅ NUM_DIFFICULTIES: {num_difficulties}")
            print(f"  ✅ Total task types: {num_families * num_difficulties}")
            return True
        else:
            print("  ⚠️  Could not find NUM_FAMILIES or NUM_DIFFICULTIES")
            return False
    except Exception as e:
        print(f"  ❌ Error checking task generator: {e}")
        return False

def check_interface_compatibility():
    """Check interface compatibility by parsing files."""
    print("\n" + "="*70)
    print("INTERFACE COMPATIBILITY CHECK")
    print("="*70)
    
    try:
        # Parse teacher interfaces
        teacher_if = Path("teacher_agent_dev/interfaces.py")
        student_if = Path("student_agent_dev/interfaces.py")
        
        if not teacher_if.exists() or not student_if.exists():
            print("  ❌ Interface files not found")
            return False
        
        # Extract Task dataclass fields
        def extract_dataclass_fields(filepath, class_name):
            content = filepath.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            yield item.target.id
        
        # Get Task fields from both
        teacher_task_fields = set(extract_dataclass_fields(teacher_if, "Task"))
        student_task_fields = set(extract_dataclass_fields(student_if, "Task"))
        
        if teacher_task_fields == student_task_fields:
            print(f"  ✅ Task dataclass fields match: {sorted(teacher_task_fields)}")
        else:
            diff = teacher_task_fields.symmetric_difference(student_task_fields)
            print(f"  ⚠️  Task fields differ: {diff}")
            print(f"     Teacher: {sorted(teacher_task_fields)}")
            print(f"     Student: {sorted(student_task_fields)}")
        
        # Get StudentState fields
        teacher_state_fields = set(extract_dataclass_fields(teacher_if, "StudentState"))
        student_state_fields = set(extract_dataclass_fields(student_if, "StudentState"))
        
        if teacher_state_fields == student_state_fields:
            print(f"  ✅ StudentState dataclass fields match: {sorted(teacher_state_fields)}")
        else:
            diff = teacher_state_fields.symmetric_difference(student_state_fields)
            print(f"  ⚠️  StudentState fields differ: {diff}")
        
        return teacher_task_fields == student_task_fields and teacher_state_fields == student_state_fields
        
    except Exception as e:
        print(f"  ❌ Error checking interfaces: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_generation_safe():
    """Test task generation without importing student_env."""
    print("\n" + "="*70)
    print("TASK GENERATION TEST")
    print("="*70)
    
    try:
        # Import only task_generator (safe)
        sys.path.insert(0, str(Path.cwd()))
        from tasks.task_generator import generate_task, NUM_FAMILIES, NUM_DIFFICULTIES
        
        print(f"  Testing with {NUM_FAMILIES} families × {NUM_DIFFICULTIES} difficulties")
        
        # Generate a few sample tasks
        for family_id in range(min(2, NUM_FAMILIES)):
            for diff_id in range(min(2, NUM_DIFFICULTIES)):
                task = generate_task(family_id, diff_id, seed=42)
                if task and hasattr(task, 'human_prompt'):
                    print(f"  ✅ Generated task: family={family_id}, diff={diff_id}")
                else:
                    print(f"  ❌ Failed to generate task: family={family_id}, diff={diff_id}")
                    return False
        
        return True
    except Exception as e:
        print(f"  ❌ Task generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("SAFE PIPELINE TEST - NO PROBLEMATIC IMPORTS")
    print("="*70)
    print()
    
    results = []
    
    # Check file structure
    results.append(("File Structure", check_file_structure()))
    
    # Check task generator
    results.append(("Task Generator Constants", check_task_generator_constants()))
    
    # Check interfaces
    results.append(("Interface Compatibility", check_interface_compatibility()))
    
    # Test task generation
    results.append(("Task Generation", test_task_generation_safe()))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {name}")
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ All compatibility checks passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} check(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

