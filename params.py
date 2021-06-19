
def optimizer_params(Model):
    param_optimizer = list(Model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    return optimizer_parameters   


def save_checkpoint(state,filename = "My_Checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state,filename)


def load_checkpoint(checkpoint):
    print("=> Loading Checkpoint")
    Model = model.BTSC()
    myModel = Model.load_state_dict(checkpoint["state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer"])
    return myModel





