# K-Nearest Neighbors - Toy Example

## The Iris Flower Problem

Imagine you're a botanist trying to identify a mysterious flower!

### Our Data (Known Flowers)

| Sepal Length | Sepal Width | Species        |
|--------------|-------------|----------------|
| 5.1          | 3.5         | Setosa         |
| 4.9          | 3.0         | Setosa         |
| 7.0          | 3.2         | Versicolor     |
| 6.4          | 3.2         | Versicolor     |
| 6.3          | 3.3         | Versicolor     |
| 6.7          | 3.0         | Virginica      |
| 6.0          | 2.7         | Virginica      |

### The Mystery Flower

You find a new flower with:
- Sepal Length: 5.5
- Sepal Width: 2.6

Is it Setosa, Versicolor, or Virginica?

### Using KNN (K=3)

**Step 1: Calculate Distance**

We calculate the distance from our mystery flower to all known flowers:

```
Mystery Flower: (5.5, 2.6)

Known Flowers:
(5.1, 3.5) - Distance: 1.01  → Setosa
(4.9, 3.0) - Distance: 0.65  → Setosa
(7.0, 3.2) - Distance: 1.52  → Versicolor
(6.4, 3.2) - Distance: 0.95  → Versicolor
(6.3, 3.3) - Distance: 0.94  → Versicolor
(6.7, 3.0) - Distance: 1.24  → Virginica
(6.0, 2.7) - Distance: 0.50  → Virginica
```

**Step 2: Find 3 Nearest Neighbors**

Sorted by distance:
1. (6.0, 2.7) - Distance: 0.50 → Virginica
2. (4.9, 3.0) - Distance: 0.65 → Setosa
3. (5.1, 3.5) - Distance: 1.01 → Setosa

**Step 3: Vote!**

- Setosa: 2 votes
- Virginica: 1 vote
- Versicolor: 0 votes

### Result: SETOSA! 🌸

### Visual Representation

```
Sepal Width
    |
 4.0 |          ★ (5.1, 3.5) Setosa
    |        ★
 3.5 |      ★
    |    
 3.0 |            ● (4.9, 3.0) Setosa
    |         ●
 2.5 |                        ◆ (5.5, 2.6) MYSTERY!
    |                     ◆
 2.0 |               ■ (6.0, 2.7) Virginica
    |
    +------------------------ Sepal Length
         4.5   5.0   5.5   6.0   6.5   7.0

★ = Setosa
● = Setosa (nearest neighbor!)
◆ = Mystery flower
■ = Virginica (nearest neighbor!)
```

### Understanding Distance

We use "Euclidean Distance" - it's like measuring with a ruler!

```
Distance = √[(x2-x1)² + (y2-y1)²]

For (5.5, 2.6) to (4.9, 3.0):
= √[(5.5-4.9)² + (2.6-3.0)²]
= √[0.6² + (-0.4)²]
= √[0.36 + 0.16]
= √0.52
= 0.72
```

### Key Takeaway

KNN finds the K closest flowers and lets them vote! It's like asking your nearest neighbors for advice!
