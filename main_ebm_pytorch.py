import torch
import numpy as np
import os
import build_data
import build_data_abstracts
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import model.dataset as dataset
import model.engine as engine
from model.modeltr import EntityModel
from model.data_utils import CoNLLDataset, data_to_dict
from model.config import Config
from model.utils import timeSince, metrics
import model.utils as utils

# create instance of config
config = Config()

data_prefix = 'p1_all'

print("Use sentences:",config.use_sen)

if config.use_sen:
    print("Dataset Contains each sentence.")
    print('Generating tagged and formatted data for the model')
    build_data.generate_model_data(data_prefix)
    print('Finished. Ready to train model on %s' %data_prefix)
    cwd = os.getcwd()
    config.filename_dev = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_dev))
    config.filename_test = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_test))
    config.filename_train = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_train))
    in_tp = 'Sentences'

else:
    print("Dataset Contains whole abstract.")
    print('Generating tagged and formatted data for the model')
    build_data_abstracts.generate_model_data(data_prefix)
    print('Finished. Ready to train model on %s' % data_prefix)
    cwd = os.getcwd()
    config.filename_dev = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_dev_abs))
    config.filename_test = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_test_abs))
    config.filename_train = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_train_abs))
    in_tp = "Abstract"


datast = 'EBM Dataset'
print(datast)
parameters = {
    'Dataset:':datast,
    'Use:':in_tp,
    'Pretrained Language Model:':Config.PRETRAINED_MODEL,
    # 'LSTM Hidden Layers:':Config.hidden_size_lstm,
    'Number of epochs:':Config.nepochs,
    'Learning Rate:':Config.lr,
    'N:':Config.N,
    'h:':Config.h,
    'k:':Config.k,
    'd_model:':Config.d_model,
    'FeedForward layers -> vanilla:':Config.d_ff,
    'FeedForward layers -> linformer:':Config.d_model*4

}

print("Parameters:\n")
for i in parameters:
    print(i,parameters[i])


# create datasets
dev = CoNLLDataset(config.filename_dev)

train = CoNLLDataset(config.filename_train)

test  = CoNLLDataset(config.filename_test)

print('Transform data to dictionaries:')
devDict = data_to_dict(dev)
trainDict = data_to_dict(train)
testDict = data_to_dict(test)

print(devDict['words'][0])
print('\n')

num_tag = len(list(utils.enc_trans.classes_))


print('Initialize dataloaders:')

train_dataset = dataset.EntityDataset(
        words=trainDict['words'], tags=trainDict['enc_tags'], config=config
    )

train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.train_batch_size, num_workers=4
    )

valid_dataset = dataset.EntityDataset(
        words=devDict['words'], tags=devDict['enc_tags'], config=config
    )

valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=Config.valid_batch_size, num_workers=1
    )

test_dataset = dataset.EntityDataset(
        words=testDict['words'], tags=testDict['enc_tags'], config=config
    )

test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=1
    )

device = torch.device("cuda")
model = EntityModel(num_tag=num_tag, tr=True)
model.to(device)
print('Model:\n', model)


param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

num_train_steps = int(len(trainDict['words']) / Config.train_batch_size * Config.nepochs)
optimizer = AdamW(optimizer_parameters, lr=Config.lr)
scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

print("Training starts...")
start = time.time()

best_loss = np.inf
for epoch in range(Config.nepochs):
    print("Epoch %g from %g"%(epoch+1,Config.nepochs))
    train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
    test_loss = engine.eval_fn(valid_data_loader, model, device)
    print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
    if test_loss < best_loss:
        torch.save(model.state_dict(), Config.MODEL_PATH)
        best_loss = test_loss
        print("New best score!")

total_time = timeSince(start)

print('Total training time:',total_time)

print("Testing...\n")

model = EntityModel(num_tag=num_tag, tr=False)
model.load_state_dict(torch.load(Config.MODEL_PATH))
model.to(device)
res = engine.test_fn(test_data_loader, model, device)


print("Labels, no X")
metrics(res['y'],res['y_pred'],  labels=utils.def_tags[:-1], target_names=utils.labels)

print("Parameters:\n")
for i in parameters:
    print(i,parameters[i])


