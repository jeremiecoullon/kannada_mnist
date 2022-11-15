import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

import pandas as pd
import numpy as np

from models import build_model
from dataset import get_train_and_validation_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")


def train(num_epochs, model_type):
    trainloader, valloader, testloader = get_train_and_validation_dataset(batch_size=128, proportion_train=0.85)

    model = build_model(model_type=model_type)

    learning_rate = 5e-3
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min')

    def train_epoch(epoch):
        model.train()
        for b_idx, (X_batch, y_batch) in enumerate(trainloader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimiser.zero_grad()
            log_probs = model(X_batch)
            loss = criterion(log_probs, y_batch)
            loss.backward()
            optimiser.step()

            pred = log_probs.data.max(1, keepdim=True)[1]
            correct = pred.eq(y_batch.data.view_as(pred)).sum()
        print(f"----\nEpoch {epoch}: training loss={loss.item():.4f}")

    def validate():
        model.eval()
        validate_loss = 0
        correct = 0

        with torch.no_grad():
            for X, y in valloader:
                X = X.to(device)
                y = y.to(device)
                log_probs = model(X)
                validate_loss += criterion(log_probs, y).item()
                pred = log_probs.data.max(1, keepdim=True)[1]
                correct += pred.eq(y.data.view_as(pred)).sum()
        validate_loss /= len(valloader.dataset)
        print(f"Validation: loss={validate_loss:.6f}, accuracy={correct/len(X):.4f}%")
        return validate_loss

    print(f"Training Kannada-MNIST using a {model_type} for {num_epochs} epochs\n")

    for epoch in range(1, num_epochs+1):
        train_epoch(epoch)
        validate_loss = validate()
        scheduler.step(validate_loss)
    print("Done.")

    # for Kaggle submission
    X_test, _ = next(iter(testloader))
    X_test = X_test.to(device)
    test_preds = model(X_test).data.max(dim=1)[1].detach().cpu().numpy()

    test_preds_df = pd.DataFrame(np.stack([np.arange(len(test_preds)), np.array(test_preds)]).T,
                                 columns=['id', 'label'])

    test_preds_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    train(num_epochs=1, model_type="mlp")
