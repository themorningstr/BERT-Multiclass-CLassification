import torch
import Dataset 
import torch
import Dataset
import config 
import engine
import params
import model
import numpy as np
import transformers
from transformers import BertConfig
from transformers import AdamW,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


device = "cuda" if torch.cuda.is_available() else "cpu"






def train():

    # Loding Training Dataset From "train.csv"
    trainTextDataset = Dataset.TextDataset(config.TRAINING_FILE)
    trainTextItem = trainTextDataset.__getitem__()

   # Loding Validation Dataset From "dev.csv"
    valTextDataset = Dataset.TextDataset(config.VALIDATION_FILE)
    valTextItem = valTextDataset.__getitem__()

   # Loading Train_DataLoader and Val_DataLoader

    Train_Dataset = Dataset.createDataset(input_id = trainTextItem["input_ids"],
                                          attention_mask = trainTextItem["attention_masks"],
                                          label = trainTextItem["targets"])

    Train_DataLoader = DataLoader(Train_Dataset,
                                  batch_size = config.TRAIN_BATCH_SIZE,
                                  sampler = RandomSampler(Train_Dataset),
                                  num_workers = 4)

    Validation_Dataset = Dataset.createDataset(input_id = valTextItem["input_ids"],
                                               attention_mask = valTextItem["attention_masks"],
                                               label = valTextItem["targets"])

    Validation_DataLoader = DataLoader(Validation_Dataset,
                                       batch_size = config.VALID_BATCH_SIZE,
                                       sampler = SequentialSampler(Validation_Dataset),
                                       num_workers = 1)
                                           

    Model_Config = transformers.BertConfig.from_pretrained("bert-base-uncased")
    Model = model.BTSC(conf = Model_Config)
    Model.to(device)

    optimizer_parameters = params.optimizer_params(Model)
    optimizer = AdamW(params = optimizer_parameters,lr = 2e-5, eps = 1e-8,weight_decay = 0.0001)

    total_steps = len(Train_DataLoader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)



    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(Train_DataLoader, Model, optimizer, device, scheduler)
        test_loss = engine.eval_fn(Validation_DataLoader, Model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(Model.state_dict(),config.MODEL_PATH)
            best_loss = test_loss


if __name__ == "__main__":
    train()