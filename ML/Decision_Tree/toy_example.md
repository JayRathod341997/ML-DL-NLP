# Decision Tree - Toy Example

## The Should I Play Tennis Problem?

Imagine you want to decide if you should play tennis today based on the weather!

### Our Data (Observations)

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

### The Decision Tree

```
                    Is Outlook = Overcast?
                    /              \
                YES/                \NO
                  /                   \
            Play Tennis!         Is Outlook = Sunny?
                                 /              \
                             YES/                \NO
                               /                   \
                        Is Humidity = High?    Is Outlook = Rainy?
                        /              \        /              \
                    YES/              \NO  YES/              \NO
                      /                \    /                \
                 DON'T Play!      Play Tennis!  Is Wind = Strong?
                                       /              \
                                   YES/              \NO
                                     /                \
                               DON'T Play!      Play Tennis!
```

### How to Read the Tree

1. **First question**: Is it overcast?
   - YES → Play tennis! (always yes)
   - NO → Keep going...

2. **If Sunny**: Is humidity high?
   - YES → Don't play (too humid!)
   - NO → Play tennis

3. **If Rainy**: Is wind strong?
   - YES → Don't play (too windy!)
   - NO → Play tennis

### Predict for New Days

**Day 1**: Outlook=Sunny, Humidity=Normal, Wind=Weak
- Overcast? NO → Sunny? YES → Humidity Normal? NO
- → **PLAY TENNIS!** ✓

**Day 2**: Outlook=Rainy, Humidity=High, Wind=Strong
- Overcast? NO → Sunny? NO → Rainy? YES → Wind Strong? YES
- → **DON'T PLAY!** ✓

### Visual Representation

```
                    ┌─────────────┐
                    │  OVERCAST?  │
                    └──────┬──────┘
                     YES/   \NO
                      │     │
                      ▼     ▼
                ┌─────────┐ ┌─────────┐
                │  PLAY   │ │ SUNNY?  │
                └─────────┘ └────┬────┘
                           YES/   \NO
                            │     │
                            ▼     ▼
                     ┌─────────┐ ┌─────────┐
                     │HUMIDITY?│ │ RAINY?  │
                     └────┬────┘ └────┬────┘
                     HIGH/   \NORM  YES/   \NO
                      │      │       │     │
                      ▼      ▼       ▼     ▼
                 ┌───────┐ ┌────┐ ┌─────┐ ┌──────┐
                 │ DON'T │ │PLAY│ │PLAY │ │WIND? │
                 │  PLAY │ └────┘ └─────┘ └──┬───┘
                 └───────┘                 STR/  \WEAK
                                          │     │
                                          ▼     ▼
                                      ┌─────┐ ┌────┐
                                      │DON'T│ │PLAY│
                                      │PLAY │ └────┘
                                      └─────┘
```

### Key Takeaway

Decision Trees break big decisions into smaller yes/no questions. Each question helps split the data into smaller groups until we can make a clear decision!
