import torch
import pandas as pd
from torch.utils.data import TensorDataset
import config




class TextDataset:
    def __init__(self,path:str):
        self.df = pd.read_csv(path,header = None,names = ["Text","Symbol","Category"])
        self.tokenizer = config.TOKENIZER


    def __len__(self):
        return len(self.df)

    def __getitem__(self):
        input_ids = []
        attention_masks = []
        possible_labels = self.df.Category.unique()

        label_dict = {}

        for index,data in enumerate(possible_labels):
            label_dict[data] = index  
        
        self.df['Targets'] = self.df.Category.replace(label_dict)
        self.data = self.df.Text.values
        self.targets = self.df.Targets

        for line in self.data:
            encoded_dict  = self.tokenizer.encode_plus(line,
                                                        add_special_tokens = True,
                                                        max_length = config.MAX_LEN,
                                                        pad_to_max_length = True,
                                                        return_attention_mask = True,
                                                        return_tensors = 'pt',
                                                        truncation=True)
        
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])

        return{
            "input_ids" : torch.cat(input_ids,dim = 0),
            "attention_masks" : torch.cat(attention_masks,dim = 0),
            "targets" : torch.tensor(self.targets,dtype = torch.int64)
        }

def createDataset(input_id,attention_mask,label):
    dataset = TensorDataset(input_id,attention_mask,label)
    return dataset



        

    
