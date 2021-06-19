import torch
import config
import params
import Dataset
import transformers


device = "cude" if torch.cuda.is_available() else "cpu"


def prediction():

    if config.LOAD_MODEL:
        Model = params.load_checkpoint(torch.load("My_Checkpoint.pth.tar"))

    # Loding Training Dataset From "train.csv"
    testTextDataset = Dataset.TextDataset(config.TESTING_FILE)
    testTextItem = testTextDataset.__getitem__()

    Test_Dataset = Dataset.createDataset(input_id = testTextItem["input_ids"],
                                               attention_mask = testTextItem["attention_masks"],
                                               label = testTextItem["targets"])

    Test_DataLoader = torch.utils.data.DataLoader(Test_Dataset,
                                                  batch_size = config.TEST_BATCH_SIZE,
                                                  sampler = torch.utils.data.SequentialSampler(Test_Dataset),
                                                  num_workers = 2)

    print('Predicting labels for {:,} test sentences...'.format(len(testTextItem["input_ids"])))

    Model.eval()


    Predictions, True_label = [],[]

    for index,batch in enumerate(Test_DataLoader):

        batch = tuple(b.to(device) for b in batch)

        Batch_input_id,Batch_attention_mask,Batch_target_label = batch

        with torch.no_grad():
            output = Model(Batch_input_id,Batch_attention_mask)

            logit = output[0]
            logit = logit.detach().cpu().numpy()
            Batch_target_label.to("cpu").numpy()
            Batch_target_label = Batch_target_label.to("cpu").numpy()


            Predictions.append(logit)
            True_label.append(Batch_target_label)

        print("Prediction Done.....!!!")



if __name__ == "__main__":
    prediction()