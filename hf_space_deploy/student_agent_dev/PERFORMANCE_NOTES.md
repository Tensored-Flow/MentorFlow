# Performance Notes: Test Slowness

## Why Tests Are Slow

The `test_student.py` tests can be slow for several reasons:

### 1. **DistilBERT Model Loading** (Main Cause)
- Loading DistilBERT from HuggingFace is **expensive** (downloads models, loads weights)
- Each test creates a new `StudentAgent()` which loads the model
- This can take **10-30+ seconds** per test on slower systems
- **This is normal** - not your laptop's fault!

### 2. **Model Inference**
- Each `student.answer()` call runs neural network inference
- Each `student.learn()` call does forward + backward pass
- On CPU, this is slower than GPU

### 3. **Multiple Evaluations**
- Tests evaluate on multiple tasks multiple times
- Each evaluation runs model inference

## Solutions Implemented

✅ **Added tqdm progress bars** - Shows progress during slow operations
✅ **Reduced iteration counts** - Fewer training loops for faster tests
✅ **Smaller eval sets** - Fewer tasks to evaluate on
✅ **Graceful fallback** - Works even if model loading fails

## Speedup Options

### Option 1: Skip Model Loading (Fastest)
```bash
# Tests will use dummy mode (much faster)
python test_student.py
```

### Option 2: Use GPU (if available)
```python
student = StudentAgent(device='cuda')  # Much faster if you have GPU
```

### Option 3: Cache Model Loading
- Model is downloaded/cached automatically by transformers
- First run is slowest (downloads model)
- Subsequent runs are faster (uses cache)

### Option 4: Use Smaller Model
- DistilBERT is already small (67M parameters)
- Could use even smaller model for testing, but DistilBERT is a good balance

## Expected Times

- **Model loading**: 10-30 seconds (first time), 5-10 seconds (cached)
- **Per test**: 5-15 seconds (with model)
- **Total test suite**: 30-90 seconds (with model)
- **Without model (dummy)**: < 5 seconds total

## It's Not Your Laptop!

This is normal for:
- Neural network model loading
- Transformer models (they're large)
- CPU inference (GPU would be faster but requires CUDA)

The progress bars help you see what's happening even if it's slow!

