import torch
from tqdm import tqdm
import numpy
import model.utils as utils

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss, _, _ = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)



def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        loss, _, _ = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)


def test_fn(data_loader, model, device):
    res = {'y_pred':[],
           'y':[]
           }

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)
            tag, target_tag = model(**data)

            predicted = utils.enc_trans.inverse_transform(tag.cpu().numpy().reshape(-1))
            target = utils.enc_trans.inverse_transform(target_tag.cpu().numpy().reshape(-1))

            res['y_pred'].append(predicted)
            res['y'].append(target)
    return res
