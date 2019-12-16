import torch
from torch.nn import functional as F

## Will be deleted in the future
class CustomOptimizer: 
    None

def train(model, train_loader, optimizer, epoch):
    model.train()
    stats = []
    for epoch_i in range(1, epoch + 1):
        for batch_idx, (input, output) in enumerate(train_loader):
            if torch.cuda.is_available():           
                input, output = input.cuda(), output.cuda()

            if isinstance(optimizer, CustomOptimizer):
                (loss, predictions) = optimizer.step(model_fn, loss_fn)
            else:
                # standard optimizer
                optimizer.zero_grad()
                predictions = model(input)
                loss = F.cross_entropy(predictions, output)
                loss.backward()
                optimizer.step()

            predictions = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            acc = predictions.eq(output.view_as(predictions)).double().mean()

            stats.append([loss.item(), acc.item()])
        
        print(f'[{epoch_i}/{epoch}] epoch: Loss: {stats[-1][0]:.3f}  Accuracy:{stats[-1][1]:.3f}')
    return stats