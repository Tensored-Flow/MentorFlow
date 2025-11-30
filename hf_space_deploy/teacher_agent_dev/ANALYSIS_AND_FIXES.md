# Analysis: Why Accuracy Drops and How to Fix

## Issue 1: Accuracy Drops at End ❌

### Root Causes Found:

1. **Evaluation uses NEW tasks each time** (line 171-175 in compare_strategies.py)
   - `general_accuracy = student.evaluate([generator.generate_task(...) for ...])`
   - Creates new tasks every iteration → variance and inconsistency
   - Should use FIXED eval set

2. **Forgetting rate too aggressive for 500 iterations**
   - Forgetting rate: 0.05
   - After 500 iterations (500 time units): retention = exp(-0.05 * 500) ≈ 0.0
   - **All skills forgotten by the end!**
   - Retention drops to near-zero after ~50-100 time units

3. **Evaluation timing confusion**
   - Currently: Learn → Evaluate → Advance time
   - Should be clearer about when evaluation happens relative to forgetting

## Issue 2: Accuracy Calculation Method

### Current Method:
- Uses `student.evaluate(eval_tasks)` which:
  - Calls `answer()` for each task (stochastic, uses randomness)
  - Accounts for forgetting via `_get_effective_skill()`
  - Returns fraction of correct answers

### Problems:
1. **Stochastic variance**: Random sampling introduces noise
2. **Eval tasks regenerated**: Different tasks each time = inconsistent
3. **Small eval set**: Only 10-15 tasks = high variance

### Better Methods:
1. **Use FIXED eval set** generated once at start
2. **Use expected accuracy** instead of sampled (less variance)
   - Expected acc = mean(prob_correct) over all tasks
3. **Larger eval set** (50-100 tasks) for stability
4. **Separate eval timing**: Evaluate BEFORE time advance

## Issue 3: Mock vs Real Components

### Current Mock Components:

**Mock Student:**
- ✅ Captures learning and forgetting
- ✅ Per-topic skill tracking
- ✅ Realistic Ebbinghaus curve
- ❌ Simplified learning model (linear skill increase)
- ❌ Stochastic but not as complex as real PPO

**Mock Task Generator:**
- ✅ Simple template-based tasks
- ✅ Multiple topics and difficulties
- ❌ Fixed templates (not procedural)
- ❌ Limited diversity

**Real Components (in MentorFlow):**
- Student: Full PPO agent with neural network
- Task Generator: Procedural generation with 15 task families

### Will Real Components Be Better?

**YES, likely:**
1. **Real PPO student** can learn more complex patterns
2. **Procedural task generator** provides more diverse tasks
3. **Better generalization** to unseen tasks
4. **More realistic learning curves**

**BUT:**
- Real components are slower to train
- Harder to debug and verify
- Teacher agent algorithm (UCB) should still work

## Recommended Fixes

1. **Fix evaluation to use FIXED eval sets**
2. **Reduce forgetting rate** or **reset time** periodically
3. **Use expected accuracy** for more stable measurements
4. **Add evaluation BEFORE time advance** option
5. **Document evaluation methodology** clearly

