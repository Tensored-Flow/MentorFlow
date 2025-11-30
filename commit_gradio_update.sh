#!/bin/bash
# Script to commit Gradio update to Hugging Face Space

echo "🔄 Updating Gradio to fix security vulnerabilities..."
echo ""

# Navigate to HF Space directory
cd /Users/leonardowang/MentorFlow/MentorFlow

# Copy updated files
echo "📋 Copying updated files..."
cp ../requirements_hf.txt requirements_hf.txt
cp ../README_HF_SPACE.md README.md
cp ../app.py app.py

echo "✅ Files copied"
echo ""

# Check git status
echo "📊 Git status:"
git status --short

echo ""
echo "📝 Ready to commit. Run these commands:"
echo ""
echo "  cd /Users/leonardowang/MentorFlow/MentorFlow"
echo "  git add requirements*.txt README.md app.py"
echo "  git commit -m 'Update Gradio to 4.44.0 to fix security vulnerabilities'"
echo "  git push"
echo ""

