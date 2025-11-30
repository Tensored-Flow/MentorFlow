#!/bin/bash
# Deployment script for Hugging Face Spaces (not used for prod/local deploys).
# If you're NOT pushing to HF Spaces, stop here—this script should be skipped.
if [ -z "$HF_DEPLOY" ]; then
  echo "⚠️  This script is only for Hugging Face Spaces. Skipping because HF_DEPLOY is not set."
  echo "    For local/prod deployment, run your normal entrypoints (app.py / compare_strategies.py) directly."
  exit 0
fi

set -e  # Exit on error

echo "🚀 Deploying MentorFlow to Hugging Face Spaces"
echo "================================================"
echo ""

# Step 1: Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Are you in the MentorFlow root directory?"
    exit 1
fi

# Step 2: Check if hf_space_deploy exists
if [ ! -d "hf_space_deploy" ]; then
    echo "❌ Error: hf_space_deploy/ directory not found."
    echo "   Run: python prepare_hf_deploy.py first"
    exit 1
fi

# Step 3: Get HF Space directory path
HF_SPACE_NAME="MentorFlow"
HF_SPACE_DIR="../${HF_SPACE_NAME}"

echo "📋 Deployment Checklist:"
echo "  1. Cloning Hugging Face Space..."
echo "  2. Copying files..."
echo "  3. Cleaning up..."
echo "  4. Committing and pushing..."
echo ""

# Check if Space directory exists
if [ -d "$HF_SPACE_DIR" ]; then
    echo "⚠️  Space directory already exists: $HF_SPACE_DIR"
    read -p "   Use existing directory? (y/n): " use_existing
    if [ "$use_existing" != "y" ]; then
        echo "   Please clone manually: git clone https://huggingface.co/spaces/iteratehack/${HF_SPACE_NAME}"
        exit 0
    fi
else
    echo "📥 Step 1: Cloning Hugging Face Space..."
    git clone https://huggingface.co/spaces/iteratehack/${HF_SPACE_NAME} "$HF_SPACE_DIR" || {
        echo "❌ Failed to clone Space. Please clone manually:"
        echo "   git clone https://huggingface.co/spaces/iteratehack/${HF_SPACE_NAME}"
        exit 1
    }
fi

cd "$HF_SPACE_DIR" || exit 1

echo ""
echo "📁 Step 2: Copying files from hf_space_deploy/..."
cp -r ../MentorFlow/hf_space_deploy/* .

echo ""
echo "🧹 Step 3: Cleaning up cache files..."
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "✅ Files ready!"
echo ""
echo "📝 Step 4: Review changes..."
git status

echo ""
read -p "Ready to commit and push? (y/n): " confirm
if [ "$confirm" = "y" ]; then
    echo ""
    echo "💾 Committing changes..."
    git add .
    git commit -m "Deploy MentorFlow with GPU support"
    
    echo ""
    echo "🚀 Pushing to Hugging Face..."
    git push
    
    echo ""
    echo "✅ Deployment complete!"
    echo ""
    echo "🌐 Your Space will be live at:"
    echo "   https://huggingface.co/spaces/iteratehack/${HF_SPACE_NAME}"
    echo ""
    echo "⏳ Wait 2-5 minutes for build to complete, then check the Logs tab."
else
    echo ""
    echo "⏸️  Stopped. You can commit manually:"
    echo "   git add ."
    echo "   git commit -m 'Deploy MentorFlow with GPU support'"
    echo "   git push"
fi
