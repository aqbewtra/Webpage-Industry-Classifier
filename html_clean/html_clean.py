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

# Model parameter
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=True, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
# fields = [('label', label_field), ('title', text_field), ('text', text_field), ('titletext', text_field)]
fields = [('label', label_field), ('text', text_field)]

# TabularDataset

train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv',
                                           test='test.csv', format='CSV', fields=fields, skip_header=True)

# Iterators

train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)