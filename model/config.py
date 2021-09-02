import os
import transformers


class Config():
    def __init__(self):
        """Initialize hyperparameters and load vocabs"""

        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)


    cwd = os.getcwd()


    # dataset sentences
    filename_dev = os.path.join(cwd,   "data/dev.txt")
    filename_test = os.path.join(cwd,  "data/gold.txt")
    filename_train = os.path.join(cwd, "data/train.txt")


    # dataset abstract
    filename_dev_abs = os.path.join(cwd,   "data/dev_abstracts.txt")
    filename_test_abs = os.path.join(cwd,  "data/gold_abstracts.txt")
    filename_train_abs = os.path.join(cwd, "data/train_abstracts.txt")

    use_sen = False # If True use sentences else use whole abstract

    max_iter = None # if not None, max number of examples in Dataset

    # Bert Model Config
    BASE_MODEL_PATH = "../Bert"
    PRETRAINED_MODEL = "allenai/scibert_scivocab_uncased"
    # PRETRAINED_MODEL = "seyonec/PubChem10M_SMILES_BPE_450k"
    # PRETRAINED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
    dir_output = os.path.join(cwd, "Results/")
    MODEL_PATH = os.path.join(dir_output, "model.bin")
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL, do_lower_case=True)
    MODEL = transformers.AutoModel.from_pretrained(PRETRAINED_MODEL)

    # Initialize special tokens

    cls_tok = '[CLS]'
    sep_tok = '[SEP]'
    pad_tok = '[PAD]'

    cls_tok = TOKENIZER.encode([cls_tok], add_special_tokens=False)
    sep_tok = TOKENIZER.encode([sep_tok], add_special_tokens=False)
    pad_tok = TOKENIZER.encode([pad_tok], add_special_tokens=False)

    class_X_enc = [4] ##EBM
    # class_X_enc = [2] ##NCBI
    # class_X_enc = [3] ##JNLPBA


    # training
    MAX_LEN = 512
    nepochs          = 6
    bert_droppout = 0.3
    dropout          = 0.5
    train_batch_size = 4
    valid_batch_size = 4

    lr_method        = "adam"
    lr               = 3e-05
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping



    # model hyperparameters
    hidden_size = 768

    # transformers high parameters
    N = 4
    h = 4
    d_model = 768
    d_ff = 2048
    trans_dropout = 0.1
    k = 512