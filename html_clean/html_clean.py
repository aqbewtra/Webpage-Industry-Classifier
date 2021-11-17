from bs4 import BeautifulSoup
import nltk
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
# print("hello world")
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import torch


test_path = "test.html"

def read_html(path):
        with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        return BeautifulSoup(text, 'lxml')

soup = read_html(test_path)
# print(nltk.clean_html(soup))
# print(soup, soup.get_text())

# tokens = nltk.tokenize.word_tokenize(soup.get_text())
# print(tokens)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')