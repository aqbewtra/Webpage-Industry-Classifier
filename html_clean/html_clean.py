from bs4 import BeautifulSoup
import nltk
print("hello world")

test_path = "/data/mus824/data/domain_pages/pages/1/87718120189c146b064d9afae3a4866be5c98fa0.html"

def read_html(path):
        with open(path, "r") as f:
                text = f.read()
        return BeautifulSoup(text, 'lxml')

soup = read_html(test_path)
# print(nltk.clean_html(soup))
# print(soup, soup.get_text())

tokens = nltk.tokenize.word_tokenize(soup)
print(tokens)
