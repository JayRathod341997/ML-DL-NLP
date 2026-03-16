# Named Entity Recognition (NER)

## Explain Like I'm 5

Imagine you're reading a story and you want to remember all the names of people, places, and things!

NER is like highlighting important things in a story:
- **People names** (like "John", "Mary") → Highlight in BLUE
- **Places** (like "Paris", "London") → Highlight in GREEN
- **Companies** (like "Google", "Apple") → Highlight in YELLOW
- **Dates** (like "Monday", "2024") → Highlight in ORANGE

Computers do this automatically to understand who, where, and when in text!

## What is NER?

Named Entity Recognition identifies and classifies named entities in text into predefined categories.

### Common Entity Types:
- PERSON (people names)
- ORG (organizations)
- LOC (locations)
- DATE (dates, times)
- MONEY (currency)
- PERCENT (percentages)

## Where Used?

1. **Search Engines**: Index content by entities
2. **Question Answering**: Find answers about specific entities
3. **Content Summarization**: Extract key entities
4. **Customer Support**: Identify products, issues
5. **Medical Records**: Extract symptoms, drugs, diseases

## Tools:
- spaCy (production)
- NLTK (basic)
- HuggingFace Transformers (state-of-art)

## Enterprise Example:
**Google News** uses NER to cluster stories about the same entities, people, and locations.
