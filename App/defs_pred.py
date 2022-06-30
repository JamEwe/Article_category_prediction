import pandas as pd
import numpy as np
import re
import string
import dill
from sklearn import feature_extraction

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
import matplotlib.cm as cm

np.random.seed(1234)

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.linear1 = nn.Linear(53798, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 600)
        self.linear4 = nn.Linear(600, 400)
        self.linear5 = nn.Linear(400, 50)
        self.linear6 = nn.Linear(50, 6)
        self.dropout = torch.nn.Dropout(0.5)
        self.batch = torch.nn.BatchNorm1d(500)
        self.batch2 = torch.nn.BatchNorm1d(600)
        
    def forward(self, x):
        y_pred = F.relu(self.linear1(x))
        y_pred = self.dropout(y_pred)
        y_pred = F.relu(self.batch(self.linear2(y_pred)))
        y_pred = self.dropout(y_pred)
        y_pred = F.relu(self.batch2(self.linear3(y_pred)))
        y_pred = self.dropout(y_pred)
        y_pred = F.relu(self.linear4(y_pred))
        y_pred = F.relu(self.linear5(y_pred))
        y_pred = self.dropout(y_pred)
        y_pred = self.linear6(y_pred)
        y_pred = torch.sigmoid(y_pred)
        return y_pred


def get_stopwords_list(stop_file_path):  
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))

def remove_stop_words(corpus):
    stopwords = get_stopwords_list('polish.txt')
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stopwords:
            if stop_word in tmp:
                tmp = [value for value in tmp if value != stop_word]
        results.append(" ".join(tmp))      
    return results

def preprocess_data(text):
    text = re.sub(r"http\S+", "", text)  
    text = text.lower()
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r' / fot. \w* \w*', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"„", '', text)
    text = re.sub(r"”", '', text)
    text = re.sub(r"\s\s+", ' ', text)
    table = str.maketrans('','',string.punctuation)
    text = text.translate(table)
    text = remove_stop_words([text])
    return text

def article_to_vector(article):
    with open('count_vectorizer', 'rb') as f:
        count_vectorizer = dill.load(f)
    vector = count_vectorizer.transform(article)
    return vector

def predict(article):
    model = FCNN()
    model.load_state_dict(torch.load('fcnn_model_cv.pt'))
    model.eval()

    article = preprocess_data(article)
    article_vector = article_to_vector(article)

    with torch.no_grad():
        data = torch.Tensor(article_vector.todense())
        output = model(data) 
    return output


def get_category(output):
    preds = [0 if i < 0 else i for i in output[0]]
    preds = torch.Tensor(preds)
    categories = ['Historia', 'Kosmos', 'Ludzie', 'Nauka', 'Odkrycia', 'Przyroda']
    text = []
    while torch.max(preds)>0:
        idx = torch.argmax(preds)
        text.append('{0}: {1:.2f}%'.format(categories[idx], preds[idx]*100))
        preds[idx] = 0.0
    if text=='':
        return ["No category"]
    else:
        return text

def get_category_more(output):
    categories = ['Historia', 'Kosmos', 'Ludzie', 'Nauka', 'Odkrycia', 'Przyroda']
    cats = []
    for o in output:
        preds = [0 if i < 0 else i for i in o]
        preds = torch.Tensor(preds)
        cat = []
        while torch.max(preds)>=0.4:
            idx = torch.argmax(preds)
            cat.append(categories[idx])
            preds[idx] = 0.0
        cats.append(cat)
    if len(cats)==0:
        return ["No category"]
    else:
        return cats


def predict_more(df_articles):
    model = FCNN()
    model.load_state_dict(torch.load('fcnn_model_cv.pt'))
    model.eval()
    #df_articles['Content']=df_articles['Content'].apply(lambda x : preprocess_data(x))

    articles_vector = article_to_vector(df_articles['Content'])

    with torch.no_grad():
        data = torch.Tensor(articles_vector.todense())
        output = model(data) 
    return output

'''

def predict_more(data_frame):
	model = FCNN()
	model.load_state_dict(torch.load('fcnn_model_cv.pt'))
	model.eval()

	graph_predictions = []

	for smiles in data_frame['smiles']:
		featurizer = GraphFeaturizer(y_column='pIC50')
		graph = featurizer(smiles)
		X, E = graph[0]
		data = Data(x=torch.FloatTensor(X), edge_index=torch.LongTensor(E))
		x, edge_index, batch = data.x, data.edge_index, data.batch
		pred = model(x, edge_index, torch.zeros(x.shape[0], dtype=torch.int64))
		pred.backward()
		graph_predictions.append(pred.data.cpu().numpy()[0][0])

	return np.array(graph_predictions) 

'''