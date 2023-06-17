import torch
import sys
import json
import langid
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, MarianMTModel, MarianTokenizer, BertTokenizer, BertForSequenceClassification

def identify_language(text):
    lang, _ = langid.classify(text)
    return lang

def translate_text(text, source_lang_code, target_lang_code):
    translator_model_name = f'Helsinki-NLP/opus-mt-{source_lang_code}-en'
    translator_model = MarianMTModel.from_pretrained(translator_model_name)
    translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
    
    encoded_input = translator_tokenizer.prepare_seq2seq_batch(src_texts=[text], return_tensors='pt')
    translated = translator_model.generate(**encoded_input)
    translated_text = translator_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    
    return translated_text

def classify_intent(user_input_text, selected_model):
    if(selected_model == 'distilbert'):
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=210)
        model.load_state_dict(torch.load('./models/massive-us-en.pt', map_location=torch.device('cpu')))
    elif(selected_model == 'bert'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=210)
        model.load_state_dict(torch.load('./models/few-shot-bert.pt', map_location=torch.device('cpu')))
    elif(selected_model == 'ensemble'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=60)
        model.load_state_dict(torch.load('./models/distilbert_bert_ensemble.pt', map_location=torch.device('cpu')))
    else:
        return "Invalid model name"
    
    lang = identify_language(user_input_text)
    if lang != 'en':
        translated_text = translate_text(user_input_text, lang, 'en')
        user_input_text = translated_text
    
    user_input = tokenizer(user_input_text, truncation=True, padding=True)
    input_ids = torch.tensor(user_input['input_ids']).unsqueeze(0)

    attention_mask = torch.tensor(user_input['attention_mask']).unsqueeze(0)
    outputs = model(input_ids, attention_mask=attention_mask)
    scores = outputs[0].detach().numpy().flatten()
    probs = np.exp(scores) / np.sum(np.exp(scores))
    if(selected_model in ['distilbert', 'bert']):
        merged = pd.read_csv('./data/merged.csv')
        _labels = merged['intent'].unique().tolist()
    else:
        merged = pd.read_csv('./data/massive-us-en.csv')
        _labels = merged['intent'].unique().tolist()

    label_probs = {}
    for i in range(len(probs)):
        label_probs[_labels[i]] = float(probs[i])

    sorted_labels = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)

    result = {
        'input': user_input_text,
        'input_language': lang,
        'translation_required': lang != 'en',
        'intent_probabilities': sorted_labels
    }

    return result

if __name__ == '__main__':
    user_input_text = sys.argv[1]
    selected_model = sys.argv[2]
    result_json = json.dumps(classify_intent(user_input_text, selected_model))
    print(result_json)


