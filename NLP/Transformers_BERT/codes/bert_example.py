# BERT Example using transformers library

from transformers import pipeline

# Sentiment analysis with BERT
classifier = pipeline("sentiment-analysis")

# Test
result = classifier("I love learning about transformers!")
print(result)

# Question Answering with BERT
qa = pipeline("question-answering")
context = "The transformer was introduced in 2017."
question = "When was the transformer introduced?"
result = qa(question=question, context=context)
print(result)

# Named Entity Recognition
ner = pipeline("ner", aggregation_strategy="simple")
result = ner("Apple is looking at buying U.K. startup for $1 billion")
print(result)
