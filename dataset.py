from glob import glob
import os
from torch.utils.data import Dataset
from bs4 import BeautifulSoup
import json

class HTML_Dataset(Dataset):
    def __init__(self, dir, label_dict_dir, scale):
        self.paths = glob(os.path.join(dir, '*.html'))
        # with open(label_dict_dir) as f:
        #     self.labels = json.loads(f.read())
            
        self.path_dict = dict()
        for i, i_path in enumerate(self.paths): #enumerate(sip(self.paths, self.labels))
            self.path_dict.update({i: i_path})

        self.data_len = len(self.paths)


    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        file_path = self.path_dict[index]

        # label = self.labels[file_path]
        # CONVERT LABEL TO INT/ONEHOT ---> Tokenize???

        with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        return BeautifulSoup(text, 'lxml').get_text(), "label"

if __name__ == "__main__":
    dataset_root = 'html_clean'
    label_dir = dataset_root + 'label-chips/'
    dataset = HTML_Dataset(dataset_root, label_dir, scale=1)
    
    text, label = dataset[0]

    print(text, label)

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoding = tokenizer.encode_plus(text, return_tensors = "pt")

    from transformers import BertForSequenceClassification
    num_classes = 10

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

    output = model(**encoding)




    print(output)