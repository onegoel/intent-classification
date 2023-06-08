import torch
import sys
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tabulate import tabulate
import json

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# massive_en = pd.read_csv('./data/massive-us-en.csv')
merged = pd.read_csv('./data/merged.csv')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=210)
model.load_state_dict(torch.load('./models/massive-us-en.pt', map_location=torch.device('cpu')))
model.eval()

def classify_intent(user_input_text, tokenizer, model):
    user_input = tokenizer(user_input_text, truncation=True, padding=True)
    input_ids = torch.tensor(user_input['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(user_input['attention_mask']).unsqueeze(0)
    outputs = model(input_ids, attention_mask=attention_mask)
    scores = outputs[0].detach().numpy().flatten()

    probs = np.exp(scores) / np.sum(np.exp(scores))

    _labels = merged['intent'].unique().tolist()

    label_probs = {}
    for i in range(len(probs)):
        label_probs[_labels[i]] = float(probs[i])  

    sorted_labels = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)

    result = {
        'input': user_input_text,
        'intent_probabilities': sorted_labels
    }

    return result

if __name__ == '__main__':
    user_input_text = sys.argv[1]
    result_json = json.dumps(classify_intent(user_input_text, tokenizer, model))
    print(result_json)