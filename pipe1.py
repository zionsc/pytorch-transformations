from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline('sentiment-analysis') # pipline object for sentiment analysis (task)

res = classifier("I've been waiting for a HuggingFace course my whole life.") # data to be interpreted

print(res)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name) # basically puts the text into a mathematical model for the machine to understand
#from_pretrained is a class method that returns a model instance from a pretrained model

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = classifier("I've been waiting for a HuggingFace course my whole life.")

print(res)

sequence = "Using a Transformer network is simple" # can also input multiple strings thru a list
res = tokenizer(sequence) # attention_mask --> 0 is to ignore, 1 is to use
print(res) # mathematical model for computer to understand

tokens = tokenizer.tokenize(sequence)
print(tokens) # tokenized version of the sequence

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids) # id version of the sequence

decoded_string = tokenizer.decode(ids)
print(decoded_string) # id back to decoded version of the sequence (string format)