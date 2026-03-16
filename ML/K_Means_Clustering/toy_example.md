# K-Means Clustering - Toy Example

## The Customer Spending Problem

Imagine you run a store and want to group your customers based on how much they spend and how often they visit!

### Our Data (Customers)

| Customer | Visits/Month | Spent ($) |
|----------|--------------|-----------|
| 1        | 1            | 200       |
| 2        | 2            | 300       |
| 3        | 1            | 250       |
| 4        | 8            | 800       |
| 5        | 10           | 900       |
| 6        | 7            | 750       |
| 7        | 4            | 450       |
| 8        | 5            | 500       |
| 9        | 3            | 350       |
| 10       | 6            | 600       |

### We Want 3 Groups (K=3)

### Step 1: Random Centroids

We randomly pick 3 starting points:
- Centroid A: (2, 200)
- Centroid B: (5, 500)
- Centroid C: (9, 850)

### Step 2: Assign Points to Nearest Centroid

```
Customer 1: (1, 200) → Nearest to A → Group A
Customer 2: (2, 300) → Nearest to A → Group A
Customer 3: (1, 250) → Nearest to A → Group A

Customer 4: (8, 800) → Nearest to C → Group C
Customer 5: (10, 900) → Nearest to C → Group C  
Customer 6: (7, 750) → Nearest to C → Group C

Customer 7: (4, 450) → Nearest to B → Group B
Customer 8: (5, 500) → Nearest to B → Group B
Customer 9: (3, 350) → Nearest to B → Group B
Customer 10: (6, 600) → Nearest to B → Group B
```

### Step 3: Move Centroids to Center of Their Groups

New Centroid A: Average of (1,200), (2,300), (1,250) = (1.3, 250)
New Centroid B: Average of (4,450), (5,500), (3,350), (6,600) = (4.5, 475)
New Centroid C: Average of (8,800), (10,900), (7,750) = (8.3, 817)

### Step 4: Repeat Until No Change!

After a few iterations, we get stable clusters:

```
Spent ($)
    |
1000 |                         ★★★ (Cluster C: VIP Customers)
    |                      ★★  ★
 800 |                   ★★     ★
    |                ★★       ★
 600 |             ★★       ★ (Cluster B: Regular Customers)
    |          ★★        ★
 400 |       ★★       ★
    |    ★★       ★
 200 | ★ ★    ★ (Cluster A: Occasional Customers)
    |
    +------------------------ Visits/Month
      1   2   3   4   5   6   7   8   9   10
```

### The Final Groups

**Cluster A - Occasional Customers** (Low spend, Low visits)
- Customers: 1, 2, 3
- Strategy: Send coupons to encourage more visits!

**Cluster B - Regular Customers** (Medium spend, Medium visits)
- Customers: 7, 8, 9, 10
- Strategy: Loyalty programs, special offers!

**Cluster C - VIP Customers** (High spend, High visits)
- Customers: 4, 5, 6
- Strategy: VIP treatment, exclusive events!

### Key Takeaway

K-Means finds natural groups in data by:
1. Starting with random centers
2. Assigning points to nearest center
3. Moving centers to the middle
4. Repeating until stable!
