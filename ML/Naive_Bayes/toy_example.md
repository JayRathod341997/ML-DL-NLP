# Naive Bayes - Toy Example

## The Weather Sports Problem

Should I play tennis today based on the weather?

### Our Training Data

| Outlook   | Humidity | Wind    | Play Tennis? |
|-----------|----------|---------|--------------|
| Sunny     | High     | Weak    | No           |
| Sunny     | High     | Strong  | No           |
| Sunny     | Normal   | Weak    | Yes          |
| Sunny     | Normal   | Strong  | Yes          |
| Overcast  | High     | Weak    | Yes          |
| Overcast  | High     | Strong  | Yes          |
| Overcast  | Normal   | Weak    | Yes          |
| Overcast  | Normal   | Strong  | Yes          |
| Rainy     | High     | Weak    | No           |
| Rainy     | High     | Strong  | No           |
| Rainy     | Normal   | Weak    | Yes          |
| Rainy     | Normal   | Strong  | No           |

### New Day to Predict

Outlook = Sunny, Humidity = High, Wind = Weak

### Step 1: Calculate Prior Probabilities

- P(Yes) = 9/14 = 64%
- P(No) = 5/14 = 36%

### Step 2: Calculate Likelihood for Each Feature

**Given YES:**
- P(Sunny|Yes) = 2/9 = 22%
- P(High|Yes) = 3/9 = 33%
- P(Weak|Yes) = 6/9 = 67%

**Given NO:**
- P(Sunny|No) = 3/5 = 60%
- P(High|No) = 4/5 = 80%
- P(Weak|No) = 2/5 = 40%

### Step 3: Apply Bayes' Theorem

```
P(Yes | Sunny, High, Weak) ∝ P(Sunny|Yes) × P(High|Yes) × P(Weak|Yes) × P(Yes)
                         = 0.22 × 0.33 × 0.67 × 0.64
                         = 0.031

P(No | Sunny, High, Weak) ∝ P(Sunny|No) × P(High|No) × P(Weak|No) × P(No)
                         = 0.60 × 0.80 × 0.40 × 0.36
                         = 0.069
```

### Step 4: Normalize

Total = 0.031 + 0.069 = 0.100

- P(Yes|Sunny,High,Weak) = 0.031/0.100 = **31%**
- P(No|Sunny,High,Weak) = 0.069/0.100 = **69%**

### Result: DON'T PLAY TENNIS! ☔

### Visual Representation

```
                    WEATHER EVIDENCE
                    
   Outlook: Sunny          Humidity: High         Wind: Weak
       ↓                       ↓                    ↓
   ┌───────┐               ┌───────┐            ┌───────┐
   │ Yes=22│               │ Yes=33│            │ Yes=67│
   │ No=60 │               │ No=80 │            │ No=40 │
   └───┬───┘               └───┬───┘            └───┬───┘
       │                      │                     │
       └──────────┬───────────┘                     │
                  ↓                                 ↓
          Multiply all YES                   Multiply all NO
          
          P(Yes|evidence) = 31%              P(No|evidence) = 69%
          
                  ↓
          ┌─────────────────┐
          │   PREDICTION:  │
          │   DON'T PLAY!  │
          └─────────────────┘
```

### Key Insight

Naive Bayes multiplies all the probabilities together (even though it assumes features are independent - which is the "naive" part!). The category with the highest probability wins!
