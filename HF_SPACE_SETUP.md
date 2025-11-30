# Hugging Face Space Setup Complete! 🚀

## ✅ What's Been Created

1. **`app.py`** - Gradio web interface for running comparisons
2. **`requirements_hf.txt`** - All dependencies for HF Spaces
3. **`README_HF_SPACE.md`** - Space metadata and description
4. **`DEPLOY_TO_HF.md`** - Detailed deployment instructions
5. **`prepare_hf_deploy.py`** - Helper script to prepare files
6. **`hf_space_deploy/`** - Pre-packaged deployment directory

## 🚀 Quick Deploy Steps

### Option 1: Use Pre-Prepared Files (Easiest)

```bash
# 1. Clone your HF Space
git clone https://huggingface.co/spaces/iteratehack/MentorFlow
cd MentorFlow

# 2. Copy files from hf_space_deploy/
cp -r ../hf_space_deploy/* .

# 3. Commit and push
git add .
git commit -m "Deploy MentorFlow with GPU support"
git push
```

### Option 2: Manual Setup

```bash
# 1. Clone your HF Space
git clone https://huggingface.co/spaces/iteratehack/MentorFlow
cd MentorFlow

# 2. Copy necessary directories
cp -r ../../teacher_agent_dev .
cp -r ../../student_agent_dev .
cp ../../app.py .
cp ../../requirements_hf.txt requirements.txt
cp ../../README_HF_SPACE.md README.md

# 3. Clean up (remove __pycache__, etc.)
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete

# 4. Commit and push
git add .
git commit -m "Deploy MentorFlow with GPU support"
git push
```

## 📋 What Gets Deployed

✅ **Included:**
- `app.py` - Gradio interface
- `teacher_agent_dev/` - Teacher agent system
- `student_agent_dev/` - LM Student system
- `requirements.txt` - All dependencies
- `README.md` - Space metadata

❌ **Excluded (auto-ignored):**
- `__pycache__/` directories
- Generated plots (`.png` files)
- Model files (`.zip` files)
- Local test files

## ⚙️ Configuration

### GPU Setup

The `README.md` includes:
```yaml
hardware: gpu-t4
```

This requests a GPU T4 Small (free tier). After first push:
1. Go to Space Settings → Hardware
2. Confirm GPU is allocated
3. If not, manually select "GPU T4 Small"

### Environment

The app automatically:
- Detects GPU availability
- Falls back to CPU if GPU unavailable
- Shows GPU status in the interface

## 🎯 Testing

After deployment:

1. **Check Space Status**
   - Visit: https://huggingface.co/spaces/iteratehack/MentorFlow
   - Wait for build to complete (check logs if errors)

2. **Test with Small Run**
   - Set iterations: 50
   - Click "Run Comparison"
   - Verify output and plot appear

3. **Full Test**
   - Set iterations: 500
   - Use GPU (cuda)
   - Monitor progress

## 📊 Expected Behavior

**With GPU:**
- Runtime: ~5-10 minutes for 500 iterations
- Faster model inference
- Faster gradient updates

**With CPU:**
- Runtime: ~15-30 minutes for 500 iterations
- Works but slower

## 🐛 Troubleshooting

### Build Fails

1. Check Space logs for error messages
2. Verify `requirements.txt` syntax
3. Ensure all imports are available

### Import Errors

1. Check that all directories are included
2. Verify `sys.path` setup in `app.py`
3. Check for missing `__init__.py` files

### GPU Not Working

1. Check Space Settings → Hardware
2. Verify `hardware: gpu-t4` in README.md
3. Check GPU status in app interface

### Timeout Issues

1. Reduce iterations for testing
2. Check Space timeout limits
3. Monitor progress in output

## 📝 Files Structure in Space

```
MentorFlow/
├── app.py                      # Gradio interface
├── README.md                   # Space metadata (with hardware: gpu-t4)
├── requirements.txt            # Dependencies
├── teacher_agent_dev/
│   ├── compare_strategies.py
│   ├── teacher_agent.py
│   ├── mock_task_generator.py
│   ├── mock_student.py
│   ├── interfaces.py
│   └── ...
└── student_agent_dev/
    ├── student_agent.py
    ├── memory_decay.py
    ├── interfaces.py
    └── ...
```

## 🔗 Access Your Space

Once deployed, your Space will be live at:
**https://huggingface.co/spaces/iteratehack/MentorFlow**

Share this link to let others run comparisons!

## 💡 Tips

- **First run takes longer** (model download, initialization)
- **Use deterministic mode** for reproducible results
- **Monitor logs** in Space logs tab for debugging
- **GPU allocation** may take time on first request

---

**Ready to deploy! Follow the steps above and your Space will be live shortly!** 🎉

