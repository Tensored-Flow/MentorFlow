# Running Comparison with LM Student

## Changes Made

Updated `compare_strategies.py` to use **LM Student (DistilBERT)** instead of MockStudentAgent for all three strategies:
- Random Strategy
- Progressive Strategy  
- Teacher Strategy

## Usage

```bash
cd teacher_agent_dev
python compare_strategies.py --iterations 500 --deterministic
```

## Notes

- **LM Student is slower** - Each iteration involves DistilBERT inference/fine-tuning
- Uses DistilBERT for multiple choice questions
- Online learning (fine-tunes on 1 task at a time)
- Memory decay using Ebbinghaus forgetting curve
- Per-topic skill tracking

## Parameters

- `learning_rate`: 5e-5 (LM fine-tuning rate)
- `retention_constant`: 80.0 (slower forgetting)
- `device`: 'cpu' (can be changed to 'cuda' if GPU available)
- `max_length`: 256 tokens
- `gradient_accumulation_steps`: 4

## Expected Runtime

With LM Student:
- **Random Strategy**: ~5-10 minutes for 500 iterations
- **Progressive Strategy**: ~5-10 minutes for 500 iterations
- **Teacher Strategy**: ~5-10 minutes for 500 iterations

**Total**: ~15-30 minutes for full comparison

## Fallback

If LM Student cannot be imported (e.g., transformers library missing), it will automatically fall back to MockStudentAgent.

