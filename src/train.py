import os
import sys
from datetime import datetime
from time import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import clip

from .model import CharsRnn


class EpochStatus:
    def __init__(self, data_loader, bar_length=10):
        self.bar_length = bar_length
        self.data_loader = data_loader

        self.forward_cnt = 0
        self.loss_sum = 0.0
        self.time_sum = 0.0
        self.start_time = time()
        self.last_step_time = time()
        # [*****************************]#
        self.loading_bar = "[" + (bar_length * ".") + "]"

    def step(self, loss):
        self.loss_sum += loss
        self.forward_cnt += 1.0

        forward_fin_time = time()

        step_time = forward_fin_time - self.last_step_time
        self.time_sum += step_time
        self.last_step_time=forward_fin_time

        avg_step_time = self.time_sum / self.forward_cnt
        time_remaining = avg_step_time * (len(self.data_loader) - self.forward_cnt)
        avg_loss = round(self.loss_sum / self.forward_cnt, 8)
        return avg_loss, self.sec2min(time_remaining)

    def get_loading_bar(self) -> str:
        finished_procedure = int((self.forward_cnt * self.bar_length) / len(self.data_loader))
        remaining_procedure = self.bar_length - finished_procedure
        return "[" + ("=" * finished_procedure) + (remaining_procedure * ".") + "]"

    def epoch_summary(self):
        total_time = time() - self.start_time
        avg_loss = round(self.loss_sum / self.forward_cnt, 8)
        return avg_loss, total_time

    @staticmethod
    def sec2min(seconds: float):
        return round(seconds / 60.0, 2)


def test(model: CharsRnn, test_loader, criterion, cuda=False, **kwargs):
    model.eval()

    hidden = model.init_hidden(test_loader.batch_size)

    test_status = EpochStatus(test_loader)
    with torch.no_grad():
        for data, labels in test_loader:
            if cuda:
                data, labels = data.cuda(), labels.cuda()
            out, hidden = model.forward(data, hidden)
            labels = labels.view(labels.shape[0] * labels.shape[1]).long()
            loss = criterion(out, labels)

            # loss*batch_size to get loss sum not the avg
            loss = (loss.item() * data.shape[0])

            avg_loss, time_remaining = test_status.step(loss)
            sys.stdout.flush()
            sys.stdout.write("\r Testing " + test_status.get_loading_bar() + "time remaining (m) = " + str(
                time_remaining) + " Avg Test_Loss=" + str(avg_loss))

        avg_test_loss, time_taken = test_status.epoch_summary()
        sys.stdout.write("\r Testing  " + test_status.get_loading_bar() + " time Taken (git m) = " + str(
            time_taken) + " Avg Test_Loss=" + str(avg_test_loss))
        sys.stdout.flush()
        return avg_test_loss, time_taken


def train(model: CharsRnn, train_loader, test_loader, epochs, lr, cuda=False, **kwargs):
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

    print("Testing before training ...")
    min_test_loss, _ = test(model, test_loader, criterion, cuda)
    print(f"Test Loss before Training={min_test_loss}")

    print("-----------------------------------------------------")

    for e in range(epochs):
        train_status = EpochStatus(train_loader)

        hidden = model.init_hidden(train_loader.batch_size)
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
            loss = (loss.item() * data.shape[0])
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            avg_loss, time_remaining = train_status.step(loss)
            sys.stdout.flush()
            sys.stdout.write("\r epoch " + str(e + 1) + train_status.get_loading_bar() + "time remaining (m) = " + str(
                time_remaining) + " Avg Train_Loss=" + str(time_remaining))

        avg_train_loss, train_time = train_status.epoch_summary()

        sys.stdout.write("\r epoch " + str(e + 1) + "] time Taken (git m) = " + str(
            train_time) + " Avg Train_Loss=" + str(avg_train_loss))
        sys.stdout.flush()
        train_losses.append(avg_train_loss)

        avg_test_loss, test_time = test(model, test_loader, criterion, cuda)
        test_losses.append(avg_test_loss)

        print(f" epoch {e + 1} train_loss ={avg_train_loss} test_loss={avg_test_loss}")

        # overfitting check and saving model weights if there is new test lost

        if avg_test_loss < min_test_loss:
            print(
                f"new minimum test loss {str(avg_test_loss)} ", end=" ")
            if "weight_saving_path" in kwargs:
                save_train_weights(model, avg_train_loss, avg_test_loss, kwargs["weight_saving_path"])
                print("achieved, model weights saved", end=" ")
            print()

            min_test_loss = avg_test_loss

        if avg_train_loss < avg_test_loss:
            print("!!!Warning Overfitting!!!")
        epoch_time_taken = test_time + train_time
        if "epoch_data_saving_path" in kwargs:
            notes = ""
            if "train_notes" in kwargs:
                notes = kwargs["train_notes"]
            save_epochs_to_csv(kwargs["epoch_data_saving_path"], avg_train_loss, len(train_loader.dataset),
                               avg_test_loss,
                               len(test_loader.dataset), epoch_time_taken, notes)
        print("-----------------------------------------------------")

    return train_losses, test_losses


def save_epochs_to_csv(csv_save_path, train_loss, no_train_rows, test_loss, no_test_rows, time_taken, train_notes=None):
    if train_notes is None:
        train_notes = ""
    date_now = datetime.now()
    if len(csv_save_path) == 0:
        full_path = "train_data.csv"
    else:
        full_path = f"{csv_save_path}/train_data.csv"
    row = [[train_loss, no_train_rows, test_loss, no_test_rows, time_taken, train_notes, date_now.strftime('%d/%m/%Y'),
            date_now.strftime('%H:%M:00')]]
    df = pd.DataFrame(row,
                      columns=["Train Loss", "no train rows", "Test Loss", "No test rows", "Time taken (M)", "Notes",
                               "Date", "Time"])

    if not os.path.exists(full_path):
        df.to_csv(full_path, index=False)
    else:
        df.to_csv(full_path, mode='a', header=False, index=False)


def save_train_weights(model, train_loss, test_loss, saving_path):
    """
    saves model weights with file name format Day_Month Hour_minute train_(train_loss) test_(test_loss)
    :param model: model object
    :param train_loss: train loss (float)
    :param test_loss: test loss (float)
    :param saving_path: the path you want to save the weights in
    :return: the full path of the saved file (saving_path+filename)
    """
    weight_file_name = f"{datetime.now().strftime('%m_%d %H_%M')} Train_({str(train_loss)[:8]}) Test_({str(test_loss)[:8]}).pt"
    full_path = f"{saving_path}/{weight_file_name}"

    torch.save(model.state_dict(), full_path)
    return full_path
