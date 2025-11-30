#!/bin/bash
# Commands to commit Gradio update to Hugging Face Space

echo "🚀 Committing Gradio update to Hugging Face Space..."
echo ""

cd /Users/leonardowang/MentorFlow/MentorFlow

echo "📋 Checking git status..."
git status --short

echo ""
echo "📦 Staging files..."
git add requirements_hf.txt requirements.txt README.md README_HF_SPACE.md app.py

echo ""
echo "💾 Committing changes..."
git commit -m "Update Gradio to 4.44.0 to fix security vulnerabilities

- Updated gradio from 4.0.0 to >=4.44.0 in requirements files
- Updated sdk_version from 4.0.0 to 4.44.0 in README.md
- Fixed indentation error in app.py"

echo ""
echo "📤 Pushing to Hugging Face..."
git push

echo ""
echo "✅ Done! Your Space will rebuild automatically with Gradio 4.44.0"

