# PCA - Toy Example

## The Student Scores Problem

Imagine you have student scores in different subjects!

### Our Data

| Student | Math | Science | English | History |
|---------|------|---------|---------|---------|
| 1       | 90   | 85      | 70      | 75      |
| 2       | 85   | 80      | 65      | 70      |
| 3       | 95   | 90      | 75      | 80      |
| 4       | 70   | 65      | 90      | 85      |
| 5       | 65   | 60      | 85      | 80      |
| 6       | 75   | 70      | 95      | 90      |

### Problem

We have 4 subjects - that's hard to visualize and understand!

### Solution: PCA!

PCA finds which subjects vary together!

### Step 1: Find Correlation

Notice:
- Math and Science are HIGHLY CORRELATED (students good at one are good at both!)
- English and History are HIGHLY CORRELATED (students good at one are good at both!)

### Step 2: Create Principal Components

**PC1 (Academic Ability - 50% of variation)**:
- Combines Math + Science + English + History
- Overall academic performance!

**PC2 (Direction - 30% of variation)**:
- Math/Science vs English/History
- STEM vs Arts倾向!

### Step 3: Reduce Dimensions

Instead of 4 subjects, we now have 2 numbers:

| Student | PC1  | PC2  |
|---------|------|------|
| 1       | 320  | 15   |
| 2       | 300  | 15   |
| 3       | 340  | 15   |
| 4       | 310  | -15  |
| 5       | 290  | -15  |
| 6       | 330  | -15  |

### Visual Representation

```
Original (4D - can't visualize!)     After PCA (2D - can plot!)

Math    Science  English  History       PC1
  │        │        │        │          │
 90──┼────85──┼────70──┼────75─►      320
  │        │        │        │          │
 85──┼────80──┼────65──┼────70─►      300
  │        │        │        │          │
 95──┼────90──┼────75──┼────80─►      340
  │        │        │        │          │
 70──┼────65──┼────90──┼────85─►      310
  │        │        │        │          │
 65──┼────60──┼────85──┼────80─►      290
  │        │        │        │          │
 75──┼────70──┼────95──┼────90─►      330

4 dimensions!                    2 dimensions!
Hard to see!                     Easy to plot!
```

### What We Learned

- Students 1, 2, 3: High PC1 (good overall)
- Students 4, 5, 6: Lower PC1 (struggling overall)

- Students 1, 2, 3: Positive PC2 (STEM方向)
- Students 4, 5, 6: Negative PC2 (Arts方向)

### Key Takeaway

PCA reduced 4 subjects to 2 main components while keeping most of the information!
- We lost some details
- But we can now visualize and understand the patterns!
