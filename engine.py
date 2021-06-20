import torch
from tqdm import tqdm



def train_fn(data_loader,model,optimizer,device,scheduler):
    model.train()
    final_loss = 0

    for index,data in enumerate(data_loader):
        Batch_input_id = data[0].to(device)
        Batch_attention_mask = data[1].to(device)
        Batch_target_label = data[2].to(device)
        optimizer.zero_grad()
        _,loss, = model(input_id = Batch_input_id,
                        attention_mask = Batch_attention_mask,
                        target_label = Batch_target_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss/len(data_loader)


def eval_fn(data_loader,model,device):
    model.eval()
    final_loss = 0

    for data in enumerate(data_loader):
        Batch_input_id = data[0].to(device)
        Batch_attention_mask = data[1].to(device)
        Batch_target_label = data[2].to(device)
        with torch.no_grad():
            _,loss, = model(input_id = Batch_input_id,
                            attention_mask = Batch_attention_mask,
                            target_label = Batch_target_label)
        final_loss += loss.item()
    return final_loss/len(data_loader)
