import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
massive_en = pd.read_csv('./data/massive-us-en.csv')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=60)
model.load_state_dict(torch.load('./models/massive-us-en.pt', map_location=torch.device('cpu')))
model.eval()

user_input = input('Enter a sentence: ')
user_input = tokenizer(user_input, truncation=True, padding=True)
input_ids = torch.tensor(user_input['input_ids']).unsqueeze(0)
attention_mask = torch.tensor(user_input['attention_mask']).unsqueeze(0)
outputs = model(input_ids, attention_mask=attention_mask)
scores = outputs[0].detach().numpy().flatten()

# convert scores to probabilities using softmax function
probs = np.exp(scores) / np.sum(np.exp(scores))

_labels = massive_en['intent'].unique().tolist()

label_probs = {}
for i in range(len(probs)):
    label_probs[_labels[i]] = probs[i]

sorted_labels = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)

print("Intents sorted by predicted accuracy:")
for intent, prob in sorted_labels:
    print(f"{intent}: {prob:.2%}")
