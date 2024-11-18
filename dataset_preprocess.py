from transformers import AutoTokenizer
from transformers import BertJapaneseTokenizer, BertForMaskedLM

import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

def tokenize_text(text):
    tokens = tokenizer.tokenize(text)
    return tokens

def tokenize_id(tokens):
    id = tokenizer.convert_tokens_to_ids(tokens)
    return id




tokenizer = AutoTokenizer.from_pretrained("colorfulscoop/sbert-base-ja")

df = pd.read_csv('./dataset.csv', usecols=['file','text'],encoding="utf-8")
df["tokenized_text"] = df["text"].apply(tokenize_text)
df["tokenized_id"] = df["text"].apply(tokenize_text).apply(tokenize_id)


df.to_csv("tokenized_dataset.csv", index=False)


print("done")

