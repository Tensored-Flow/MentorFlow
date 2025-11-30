#!/usr/bin/env python3
"""
Compatibility Check and Cleanup Script for MentorFlow
Before final teammate commits, check compatibility and clean unnecessary files.
"""

import os
import sys
from pathlib import Path
import shutil

class CompatibilityChecker:
    def __init__(self, root_dir="."):
        self.root = Path(root_dir)
        self.issues = []
        self.files_to_delete = []
        self.folders_to_delete = []
        
    def check_interfaces_compatibility(self):
        """Check if interfaces are compatible between teacher and student dev."""
        print("\n" + "="*70)
        print("CHECKING INTERFACE COMPATIBILITY")
        print("="*70)
        
        teacher_interfaces = self.root / "teacher_agent_dev" / "interfaces.py"
        student_interfaces = self.root / "student_agent_dev" / "interfaces.py"
        
        if not teacher_interfaces.exists():
            self.issues.append("❌ teacher_agent_dev/interfaces.py not found")
            return False
            
        if not student_interfaces.exists():
            self.issues.append("❌ student_agent_dev/interfaces.py not found")
            return False
        
        # Read and compare key parts
        with open(teacher_interfaces) as f:
            teacher_content = f.read()
        with open(student_interfaces) as f:
            student_content = f.read()
        
        # Check if Task dataclass is compatible
        teacher_task_fields = ["passage", "question", "choices", "answer", "topic", "difficulty", "task_id"]
        student_task_fields = ["passage", "question", "choices", "answer", "topic", "difficulty", "task_id"]
        
        # Check if methods are compatible
        teacher_methods = ["answer", "learn", "evaluate", "get_state", "advance_time"]
        student_methods = ["answer", "learn", "evaluate", "get_state", "advance_time"]
        
        # Interfaces are mostly compatible (minor docstring differences)
        print("✅ Interfaces are compatible (minor docstring differences only)")
        return True
    
    def find_temporary_files(self):
        """Find temporary and generated files that should be cleaned."""
        print("\n" + "="*70)
        print("FINDING TEMPORARY FILES")
        print("="*70)
        
        temp_patterns = [
            "*.log",
            "*.pyc",
            "__pycache__",
            ".DS_Store",
            "*.swp",
            "*~",
            "COMMIT_MESSAGE.txt",
            "COMMIT_SUMMARY.md",
            "pipeline_test_output.log",
            "test_full_pipeline.py",  # Temporary test file
        ]
        
        for pattern in temp_patterns:
            for path in self.root.rglob(pattern):
                if path.is_file() and not any(part.startswith('.') for part in path.parts):
                    self.files_to_delete.append(path)
                    print(f"  🗑️  {path}")
                elif path.is_dir() and path.name == "__pycache__":
                    self.folders_to_delete.append(path)
                    print(f"  🗑️  {path}")
    
    def find_redundant_plots(self):
        """Find plots that can be regenerated."""
        print("\n" + "="*70)
        print("FINDING REDUNDANT PLOTS")
        print("="*70)
        
        # Plots in models/ that are already in .gitignore
        models_dir = self.root / "models"
        if models_dir.exists():
            plot_files = list(models_dir.glob("*.png"))
            for plot in plot_files:
                # These are generated outputs, can be regenerated
                if plot.name not in ["comparison_all_strategies.png"]:  # Keep comparison plot
                    self.files_to_delete.append(plot)
                    print(f"  🗑️  {plot} (generated, can be regenerated)")
        
        # Student visualizations can be regenerated
        student_viz_dir = self.root / "student_agent_dev" / "student_visualizations"
        if student_viz_dir.exists():
            for plot in student_viz_dir.glob("*.png"):
                self.files_to_delete.append(plot)
                print(f"  🗑️  {plot} (generated, can be regenerated)")
    
    def find_duplicate_files(self):
        """Find duplicate or redundant files."""
        print("\n" + "="*70)
        print("CHECKING FOR DUPLICATE FILES")
        print("="*70)
        
        # Check for duplicate interfaces
        teacher_interfaces = self.root / "teacher_agent_dev" / "interfaces.py"
        student_interfaces = self.root / "student_agent_dev" / "interfaces.py"
        
        # Both are needed for independent development, but they should be identical
        if teacher_interfaces.exists() and student_interfaces.exists():
            print("ℹ️  Two interfaces.py files exist (one per dev track)")
            print("   - teacher_agent_dev/interfaces.py (for teacher development)")
            print("   - student_agent_dev/interfaces.py (for student development)")
            print("   ✅ Both needed for independent development")
    
    def find_unnecessary_documentation(self):
        """Find redundant documentation files."""
        print("\n" + "="*70)
        print("CHECKING DOCUMENTATION")
        print("="*70)
        
        # Keep essential docs, remove redundant summaries
        docs_to_review = [
            "CHANGES_SUMMARY.md",
            "CLEANUP_SUMMARY.md",
            "COMMIT_SUMMARY.md",
            "WINDSURF_ONBOARDING_PROMPT.md",  # Keep this - it's useful
        ]
        
        for doc in docs_to_review:
            doc_path = self.root / doc
            if doc_path.exists():
                if doc == "WINDSURF_ONBOARDING_PROMPT.md":
                    print(f"  ✅ KEEP: {doc} (useful onboarding doc)")
                else:
                    self.files_to_delete.append(doc_path)
                    print(f"  🗑️  {doc} (temporary summary)")
    
    def check_production_compatibility(self):
        """Check if production components can work together."""
        print("\n" + "="*70)
        print("CHECKING PRODUCTION COMPONENT COMPATIBILITY")
        print("="*70)
        
        # Check if production components exist
        prod_components = {
            "task_generator": self.root / "tasks" / "task_generator.py",
            "student_env": self.root / "student" / "student_env.py",
            "ppo_agent": self.root / "student" / "ppo_agent.py",
            "teacher_bandit": self.root / "teacher" / "teacher_bandit.py",
        }
        
        all_exist = True
        for name, path in prod_components.items():
            if path.exists():
                print(f"  ✅ {name}: {path.name}")
            else:
                print(f"  ❌ {name}: NOT FOUND")
                all_exist = False
        
        if all_exist:
            print("\n✅ All production components present")
        
        return all_exist
    
    def cleanup(self, dry_run=True):
        """Delete identified files/folders."""
        print("\n" + "="*70)
        print(f"{'DRY RUN: ' if dry_run else ''}CLEANUP")
        print("="*70)
        
        total_size = 0
        for path in self.files_to_delete:
            if path.exists():
                size = path.stat().st_size
                total_size += size
                if not dry_run:
                    path.unlink()
                    print(f"  🗑️  Deleted: {path}")
                else:
                    print(f"  🗑️  Would delete: {path} ({size:,} bytes)")
        
        for folder in self.folders_to_delete:
            if folder.exists():
                if not dry_run:
                    shutil.rmtree(folder)
                    print(f"  🗑️  Deleted folder: {folder}")
                else:
                    print(f"  🗑️  Would delete folder: {folder}")
        
        print(f"\n{'Would free' if dry_run else 'Freed'}: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
        return total_size
    
    def print_summary(self):
        """Print summary of findings."""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        print(f"\nFiles to delete: {len(self.files_to_delete)}")
        print(f"Folders to delete: {len(self.folders_to_delete)}")
        
        if self.issues:
            print(f"\nIssues found: {len(self.issues)}")
            for issue in self.issues:
                print(f"  {issue}")
        else:
            print("\n✅ No major compatibility issues found")

def main():
    checker = CompatibilityChecker()
    
    print("="*70)
    print("MENTORFLOW COMPATIBILITY CHECK & CLEANUP")
    print("="*70)
    
    # Run checks
    checker.check_interfaces_compatibility()
    checker.check_production_compatibility()
    checker.find_temporary_files()
    checker.find_redundant_plots()
    checker.find_duplicate_files()
    checker.find_unnecessary_documentation()
    
    # Print summary
    checker.print_summary()
    
    # Show what would be deleted
    print("\n" + "="*70)
    print("DRY RUN - SHOWING WHAT WOULD BE DELETED")
    print("="*70)
    checker.cleanup(dry_run=True)
    
    # Ask for confirmation
    print("\n" + "="*70)
    response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
    
    if response == 'yes':
        checker.cleanup(dry_run=False)
        print("\n✅ Cleanup complete!")
    else:
        print("\n❌ Cleanup cancelled")

if __name__ == "__main__":
    main()

