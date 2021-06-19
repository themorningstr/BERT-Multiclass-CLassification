import transformers

MAX_LEN = 150
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 12
EPOCHS = 10
TRAINING_FILE = "../DATASET/train.csv"
VALIDATION_FILE = "../DATASET/dev.csv"
TESTING_FILE = "../DATASET/test.csv"
MODEL_PATH = "../Saved_Model/"
NUMBER_OF_CLASSES = 151
TOKENIZER = transformers.BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
LOAD_MODEL = Ture

    


