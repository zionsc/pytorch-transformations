from transformers import pipeline

classifier = pipeline('sentiment-analysis') # pipline object for sentiment analysis (task)

res = classifier("I've been waiting for a HuggingFace course my whole life.") # data to be interpreted

print(res)