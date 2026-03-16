# Linear Regression - Toy Example

## The Ice Cream Shop Problem

Imagine you own an ice cream shop. You want to know how many ice creams you'll sell based on the temperature outside!

### Our Data (Real Observations)

| Temperature (°C) | Ice Creams Sold |
|------------------|-----------------|
| 20               | 40              |
| 25               | 55              |
| 30               | 80              |
| 35               | 110             |
| 40               | 145             |

### What We Want to Predict

If tomorrow's temperature is 38°C, how many ice creams will we sell?

### The Answer

Using Linear Regression, we find that:
- Our line is: `Ice Creams = 4.5 × Temperature - 50`
- For 38°C: Ice Creams = 4.5 × 38 - 50 = 171 - 50 = **121 ice creams**

### Visual Representation

```
Ice Creams
Sold |
145  |                    * (40°C, 145)
110  |               * (35°C, 110)
80   |            * (30°C, 80)
55   |       * (25°C, 55)
40   |   * (20°C, 40)
      +------------------------ Temperature (°C)
       20   25   30   35   40

The * are real data points
The line is our prediction
```

### Fun Exercise!

Try to guess what happens at 10°C:
- Using our formula: Ice Creams = 4.5 × 10 - 50 = 45 - 50 = -5

Wait, negative ice creams?! 😄

This shows a limitation - our model only works well within the range of data we have (20-40°C). Outside this range, predictions might not make sense!

### Key Takeaway

Linear Regression helps us find the relationship between two things so we can make predictions. It's like finding a pattern in the dots!
