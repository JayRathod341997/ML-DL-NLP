# NER Example with spaCy

import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# Text
text = "Apple is looking at buying U.K. startup for $1 billion"

# Process
doc = nlp(text)

# Extract entities
print("Entities:")
for ent in doc.ents:
    print(f"  {ent.text} -> {ent.label_}")
