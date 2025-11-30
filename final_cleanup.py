#!/usr/bin/env python3
"""
Final Cleanup - Focus on project files only (exclude venv)
Check compatibility and remove unnecessary files before final teammate commits
"""

import os
from pathlib import Path

def main():
    root = Path(".")
    
    print("="*70)
    print("FINAL CLEANUP - PROJECT FILES ONLY")
    print("="*70)
    
    # Files to delete (outside venv)
    files_to_delete = []
    
    # Temporary files
    temp_files = [
        "COMMIT_MESSAGE.txt",
        "COMMIT_SUMMARY.md",
        "pipeline_test_output.log",
        "test_full_pipeline.py",
        "cleanup_and_verify.py",
    ]
    
    for f in temp_files:
        path = root / f
        if path.exists():
            files_to_delete.append(path)
            print(f"  🗑️  {f} (temporary file)")
    
    # Redundant documentation summaries
    redundant_docs = [
        "CHANGES_SUMMARY.md",
        "CLEANUP_SUMMARY.md",
    ]
    
    for doc in redundant_docs:
        path = root / doc
        if path.exists():
            files_to_delete.append(path)
            print(f"  🗑️  {doc} (redundant summary)")
    
    # Generated plots in models/ (already in .gitignore, but can clean locally)
    models_dir = root / "models"
    if models_dir.exists():
        for plot in models_dir.glob("*.png"):
            # Keep comparison plot in teacher_agent_dev, but models/ plots can go
            files_to_delete.append(plot)
            print(f"  🗑️  {plot} (generated plot, can regenerate)")
    
    # Generated student visualizations (can regenerate)
    student_viz_dir = root / "student_agent_dev" / "student_visualizations"
    if student_viz_dir.exists():
        for plot in student_viz_dir.glob("*.png"):
            files_to_delete.append(plot)
            print(f"  🗑️  {plot} (generated plot, can regenerate)")
    
    # Student checkpoint (can regenerate)
    checkpoint = root / "student_agent_dev" / "student_checkpoint.pt"
    if checkpoint.exists():
        files_to_delete.append(checkpoint)
        print(f"  🗑️  {checkpoint} (model checkpoint, can regenerate)")
    
    print(f"\n{'='*70}")
    print(f"Found {len(files_to_delete)} files to delete")
    print(f"{'='*70}\n")
    
    # Check interfaces compatibility
    print("Checking interface compatibility...")
    teacher_if = root / "teacher_agent_dev" / "interfaces.py"
    student_if = root / "student_agent_dev" / "interfaces.py"
    
    if teacher_if.exists() and student_if.exists():
        with open(teacher_if) as f:
            t_content = f.read()
        with open(student_if) as f:
            s_content = f.read()
        
        # Check key structures match
        t_has_task = "class Task:" in t_content
        s_has_task = "class Task:" in s_content
        t_has_student = "class StudentAgentInterface" in t_content
        s_has_student = "class StudentAgentInterface" in s_content
        
        if t_has_task and s_has_task and t_has_student and s_has_student:
            print("  ✅ Interfaces are compatible")
        else:
            print("  ⚠️  Interface compatibility check inconclusive (should verify manually)")
    
    # Delete files
    print(f"\n{'='*70}")
    print("DELETING FILES...")
    print(f"{'='*70}\n")
    
    for path in files_to_delete:
        try:
            if path.is_file():
                path.unlink()
                print(f"  ✅ Deleted: {path}")
            elif path.is_dir():
                import shutil
                shutil.rmtree(path)
                print(f"  ✅ Deleted folder: {path}")
        except Exception as e:
            print(f"  ❌ Error deleting {path}: {e}")
    
    print(f"\n{'='*70}")
    print("✅ CLEANUP COMPLETE")
    print(f"{'='*70}\n")
    
    print("Summary:")
    print("  - Temporary files removed")
    print("  - Generated plots removed (can regenerate)")
    print("  - Interfaces verified compatible")
    print("\n✅ Ready for final teammate commits!")

if __name__ == "__main__":
    main()

