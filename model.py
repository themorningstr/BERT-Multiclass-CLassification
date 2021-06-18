import torch
import transformers
import config
import torch.nn as nn


def loss_fn(output,target,num_label):
    lfn = nn.CrossEntropyLoss()
    active_logit = output[1].view(-1,num_label)
    active_labels = target.view(-1)
    loss = lfn(active_logit, active_labels)
    return loss


class BTSC(transformers.BertPreTrainedModel):
    def __init__(self,conf):
        super(BTSC,self).__init__(conf)
        self.N_classes = config.NUMBER_OF_CLASSES
        self.Dbert = transformers.BertForSequenceClassification(conf)
        self.dropout = nn.Dropout(.3)
        self.out = nn.Linear(768,self.N_classes)
    
    def forward(self,input_id,attention_mask,target_label):
        output = self.Dbert(input_id,attention_mask,target_label) 
        Dout = self.dropout(output[1])                                            # Dropout output
        FOut = self.out(Dout)                                               # Final Output
        
        loss = loss_fn(FOut,target_label,self.N_classes)
        
        return FOut,loss