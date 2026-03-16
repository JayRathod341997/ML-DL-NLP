# Logistic Regression - Toy Example

## The Will I Pass The Exam Problem?

Imagine you're a student and you want to know if you'll pass or fail an exam based on how many hours you study!

### Our Data (Real Observations)

| Hours Studied | Passed Exam? |
|---------------|--------------|
| 1             | No (0)       |
| 2             | No (0)       |
| 3             | No (0)       |
| 4             | No (0)       |
| 5             | Yes (1)      |
| 6             | Yes (1)      |
| 7             | Yes (1)      |
| 8             | Yes (1)      |

### What We Want to Predict

If you study for 5.5 hours, will you pass?

### The Answer

Using Logistic Regression:
- Probability of passing at 5.5 hours: ~52%
- Prediction: **PASS** (because 52% > 50%)

### Visual Representation

```
Probability
of Passing|
    1.0   |      ___________  (High chance of passing)
          |     /
    0.5   |____/_______  (The decision boundary - 50%)
          |     \
    0.0   |      \__  (Low chance of passing)
          +------------------------ Hours Studied
           1   2   3   4   5   6   7   8

★ = Student who passed (1)
○ = Student who failed (0)
```

### Understanding the Sigmoid Curve

The curve shows:
- At 1-2 hours: Very low chance (~10%) of passing
- At 3-4 hours: Low chance (~30%) of passing
- At 5 hours: About 50% chance (the boundary!)
- At 6-7 hours: High chance (~70-90%) of passing
- At 8+ hours: Very high chance (~95%+) of passing

### Fun Exercise!

What if you study for 0 hours?
- Probability = almost 0% (you'll definitely fail!)

What if you study for 10 hours?
- Probability = almost 100% (you'll definitely pass!)

### Key Insight

The magic number is 50%!
- If probability > 50% → Predict YES (pass)
- If probability < 50% → Predict NO (fail)

This is called the "decision boundary" - it's like the line between passing and failing!
