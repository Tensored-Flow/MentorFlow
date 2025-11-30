# Summary of Fixes for Accuracy Drop Issues

## Issues Identified

### 1. **Accuracy Drops at End** ❌

**Root Causes:**
1. **Evaluation uses NEW tasks each iteration** → Variance and inconsistency
   - Line 171-175: Generates new tasks on-the-fly for `general_accuracy`
   - Different tasks each time = different difficulty/variance

2. **Forgetting rate too aggressive for 500 iterations**
   - Forgetting rate = 0.05
   - After 500 time units: retention = exp(-0.05 * 500) ≈ 0.0
   - **All skills completely forgotten by iteration 500!**

3. **Evaluation timing**: Evaluation happens after time advance, but we log before - this is actually OK

**Fix:**
- ✅ Use **FIXED eval sets** generated once at start
- ✅ Reduce forgetting rate from 0.05 to 0.01 (5x slower forgetting)
- ✅ Evaluation happens BEFORE time advance (accurate snapshot)

### 2. **Accuracy Calculation Method**

**Current Method:**
- Uses `student.evaluate(eval_tasks)` which samples answers stochastically
- Accounts for forgetting correctly
- BUT: Uses different tasks each time

**Problems:**
- Stochastic variance (random sampling)
- Inconsistent eval sets (regenerated each time)
- Small eval sets (10-15 tasks) = high variance

**Better Method:**
- ✅ **FIXED eval sets** generated once
- ✅ Same tasks used throughout = consistent measurement
- ✅ Larger eval sets (15+ tasks) for stability

**Alternative (for future):**
- Use expected accuracy = mean(prob_correct) instead of sampling
- Removes stochastic variance

### 3. **Mock vs Real Components**

**Current Mock Components:**
- ✅ Mock Student: Captures learning + forgetting well
- ✅ Mock Task Generator: Simple but functional
- ❌ Simplified learning model
- ❌ Limited task diversity

**Real Components (MentorFlow):**
- Real Student: Full PPO with neural network
- Real Task Generator: Procedural generation, 15 families

**Will Real Components Be Better?** **YES:**

1. **Real PPO Student:**
   - Can learn complex patterns
   - Better generalization
   - More realistic learning curves
   - But: Slower to train

2. **Real Task Generator:**
   - More diverse tasks
   - Procedural generation = infinite variety
   - Better tests generalization

3. **Teacher Agent Algorithm:**
   - UCB algorithm will work the same
   - Should perform even better with real components
   - More realistic reward signals

**Expected Improvement:**
- Teacher should learn better curriculum
- Student should achieve higher accuracy
- More realistic forgetting patterns (if implemented)

## Applied Fixes

✅ **Fixed evaluation to use FIXED eval sets**
✅ **Reduced forgetting rate from 0.05 → 0.01**
✅ **Evaluation happens BEFORE time advance**
✅ **All strategies use consistent eval sets**

## Remaining Considerations

1. **Forgetting Model**: Could use more sophisticated model (spaced repetition optimization)
2. **Evaluation Method**: Could use expected accuracy instead of sampling
3. **Eval Set Size**: Could increase for more stability (currently 15 tasks, could be 50-100)
4. **Time Reset**: Could periodically reset time to prevent complete forgetting in long training

