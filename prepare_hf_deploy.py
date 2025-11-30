#!/usr/bin/env python3
"""
Helper script to prepare files for Hugging Face Spaces deployment.
Copies necessary files and creates proper structure.
"""

import shutil
from pathlib import Path
import os

# HF-only export; not used for local/prod deploys.
HF_SPACE_DIR = Path("hf_space_deploy")
REQUIRED_DIRS = [
    "teacher_agent_dev",
    "student_agent_dev"
]

REQUIRED_FILES = [
    "app.py",
    "requirements_hf.txt",
    "README_HF_SPACE.md"
]

def prepare_deployment():
    """Prepare files for HF Space deployment."""
    print("=" * 70)
    print("PREPARING HUGGING FACE SPACE DEPLOYMENT")
    print("=" * 70)
    
    # Create deployment directory
    if HF_SPACE_DIR.exists():
        print(f"\n⚠️  Directory {HF_SPACE_DIR} already exists.")
        response = input("   Delete and recreate? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(HF_SPACE_DIR)
        else:
            print("   Keeping existing directory.")
    
    HF_SPACE_DIR.mkdir(exist_ok=True)
    
    # Copy required directories
    print("\n📁 Copying directories...")
    for dir_name in REQUIRED_DIRS:
        src = Path(dir_name)
        if not src.exists():
            print(f"   ❌ {dir_name} not found! Skipping...")
            continue
        
        dst = HF_SPACE_DIR / dir_name
        if dst.exists():
            shutil.rmtree(dst)
        
        print(f"   Copying {dir_name}/...")
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
            '__pycache__',
            '*.pyc',
            '*.png',  # Don't copy generated plots
            '*.zip',  # Don't copy model files
        ))
        print(f"   ✅ {dir_name}/ copied")
    
    # Copy required files
    print("\n📄 Copying files...")
    for file_name in REQUIRED_FILES:
        src = Path(file_name)
        if not src.exists():
            print(f"   ❌ {file_name} not found! Skipping...")
            continue
        
        dst = HF_SPACE_DIR / file_name
        shutil.copy2(src, dst)
        print(f"   ✅ {file_name} copied")
        
        # Special handling for requirements_hf.txt -> requirements.txt
        if file_name == "requirements_hf.txt":
            requirements_dst = HF_SPACE_DIR / "requirements.txt"
            shutil.copy2(src, requirements_dst)
            print(f"   ✅ requirements.txt created (from requirements_hf.txt)")
        
        # Special handling for README_HF_SPACE.md -> README.md
        if file_name == "README_HF_SPACE.md":
            readme_dst = HF_SPACE_DIR / "README.md"
            shutil.copy2(src, readme_dst)
            print(f"   ✅ README.md created (from README_HF_SPACE.md)")
    
    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\n📦 Files ready in: {HF_SPACE_DIR.absolute()}")
    print("\n📝 Next steps:")
    print(f"   1. cd {HF_SPACE_DIR}")
    print("   2. Copy contents to your HF Space directory")
    print("   3. Or use: git clone https://huggingface.co/spaces/iteratehack/MentorFlow")
    print("   4. Copy files and commit: git add . && git commit -m 'Deploy' && git push")
    print("\n💡 See DEPLOY_TO_HF.md for detailed instructions")

if __name__ == "__main__":
    prepare_deployment()
