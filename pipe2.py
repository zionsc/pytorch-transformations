from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2") 
#specific model distilgpt2 either locally or from model huh

res = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

print(res)