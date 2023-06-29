from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
#step by step process of pipe1.py w/ pytorch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

X_train = ["I've been waiting for a HuggingFace course my whole life.",
           "Python is great!"] # can use many sentences with this list

res = classifier(X_train) # using the specified models in classifier, predict the sentiment of X_train
print(res)

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt") # pt is pytorch format
print(batch) #truncation means to delete out of bounds, usually tokenizer() is used instead of sepearte functions used in pipe1

with torch.no_grad(): #inference in pytorch
    outputs = model(**batch) # ** --> unpacks the dictionary
    print(outputs) # outputs is a tuple of 2 tensors

    predictions = F.softmax(outputs.logits, dim=1) # softmax is a function that converts logits to probabilities
    print(predictions) # predictions is a tensor of probabilities

    labels = torch.argmax(predictions, dim=1)
    print(labels) # labels is a tensor of the predicted labels


