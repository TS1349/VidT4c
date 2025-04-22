import torch.optim as optim


optimizer = {
    "optimzer" : optim.SGD,
    "lr" : 0.005,
    "scheduler" : lambda x : x,
    "momentum" : 0.9,
    "epochs" : 15,
}



