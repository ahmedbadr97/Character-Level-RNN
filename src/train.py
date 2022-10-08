import torch
import torch.nn as nn
import torch.optim as optim

from .model import CharsRnn
from traintracker.tracker import TrainTracker


def test(model: CharsRnn, test_loader, criterion, train_tracker: TrainTracker, cuda=False, **kwargs):
    model.eval()

    hidden = model.init_hidden(test_loader.batch_size)
    train_tracker.valid()

    avg_test_loss = 0
    with torch.no_grad():
        for data, labels in test_loader:
            if cuda:
                data, labels = data.cuda(), labels.cuda()
            out, hidden = model.forward(data, hidden)
            labels = labels.view(labels.shape[0] * labels.shape[1]).long()
            loss = criterion(out, labels)

            # loss*batch_size to get loss sum not the avg
            loss = loss.item()

            avg_test_loss = train_tracker.step(loss)

        return avg_test_loss


def train(model: CharsRnn, train_loader, test_loader, epochs, lr, train_data_save_path=None, cuda=False, **kwargs):
    if "criterion" in kwargs:
        criterion = kwargs["criterion"]
    else:
        criterion = nn.CrossEntropyLoss()
    if "optimizer" in kwargs:
        optimizer = kwargs["optimizer"]
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if cuda:
        model.cuda()

    train_losses = []
    test_losses = []

    train_data_saver = None
    extra_info = {"sequence length": train_loader.dataset.seq_length}
    train_tracker = TrainTracker(model, test_loader,train_loader,  criterion, optimizer, train_data_save_path,
                                 weights_dir=kwargs["weight_saving_path"], extra_info=extra_info)

    print("Testing before training ...")
    test_only_tracker = TrainTracker(model, test_loader)
    min_test_loss = test(model, test_loader, criterion, test_only_tracker, cuda)
    test_only_tracker.end_epoch()
    print(f"Test Loss before Training={min_test_loss}")
    print("-----------------------------------------------------")
    for e in range(epochs):

        hidden = model.init_hidden(train_loader.batch_size)
        model.train()
        train_tracker.train()

        for data, labels in train_loader:
            hidden = tuple([each.data for each in hidden])
            optimizer.zero_grad()
            if cuda:
                data, labels = data.cuda(), labels.cuda()

            out, hidden = model.forward(data, hidden)

            # out shape is (batch_size*seq_size , no_chars)
            # vector of batch_size*sequence_size
            labels = labels.view(labels.shape[0] * labels.shape[1]).long()
            loss = criterion(out, labels)

            loss.backward()
            # loss * batch_size to get loss sum not the avg
            loss = loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            avg_loss = train_tracker.step(loss)

        avg_test_loss = test(model, test_loader, criterion, train_tracker, cuda)

        avg_train_loss, avg_test_loss = train_tracker.end_epoch()

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        print("-----------------------------------------------------")

    return train_losses, test_losses
