import tqdm
import torch
import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
# FIXME should I keep that?
from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
)

import matplotlib.pyplot as plt


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    # get training data from the .jason file
    # tokenize it: preprocess strings, do I do anything else?
    # split into train and validation set, the json has them labled already
    # create dataloaders for each... what on earth does that mean?

    # extracting data from the json file
    file = open('lang_to_sem_data.json')
    langToSem = json.load(file)
    trainUntoken = langToSem["train"]
    valUntoken = langToSem["valid_seen"]

    train2d = list()
    val2d = list()

    # cleaning up the commands and making this 2D instead of 3D
    for bigSet in trainUntoken:
        for smallSet in bigSet:
            temp = preprocess_string(smallSet[0])
            newList = [temp, smallSet[1]]
            train2d.append(newList)
    for bigSet in valUntoken:
        for smallSet in bigSet:
            temp = preprocess_string(smallSet[0])
            newList = [temp, smallSet[1]]
            val2d.append(newList)

    # making token table
    (vocab2Indx, indx2Vocab, maxLen) = build_tokenizer_table(trainUntoken)

    # Encoding train
    n_lines = len(train2d)
    trainEncoded = np.zeros((n_lines, maxLen), dtype=np.int32)
    idx = 0
    for example in train2d:  # goes over each list of command, meaning
        trainEncoded[idx][0] = vocab2Indx["<start>"]
        jdx = 1
        wordList = example[0].split()
        for word in wordList:  # the first element is the command
            word = word.lower()
            # if you want to do character by character like I did at first, have for word in example[0]
            if len(word) > 0:  # it is something to look at
                if word in vocab2Indx:  # important enough to have a code for it
                    trainEncoded[idx][jdx] = vocab2Indx[word]
                else:  # not important enough
                    trainEncoded[idx][jdx] = vocab2Indx["<unk>"]
                jdx += 1
                if jdx == maxLen - 1:  # we have done enough encoding for now
                    break
        trainEncoded[idx][jdx] = vocab2Indx["<end>"]
        idx += 1

    trainDS = torch.utils.data.TensorDataset(torch.from_numpy(trainEncoded))
    # Encoding val
    n_lines = len(val2d)
    valEncoded = np.zeros((n_lines, maxLen), dtype=np.int32)
    idx = 0

    for example in val2d:  # goes over each list of command, meaning
        valEncoded[idx][0] = vocab2Indx["<start>"]
        jdx = 1
        wordList = example[0].split()
        for word in wordList:  # the first element is the command
            word = word.lower()
            if len(word) > 0:  # it is something to look at
                if word in vocab2Indx:  # important enough to have a code for it
                    valEncoded[idx][jdx] = vocab2Indx[word]
                else:  # not important enough
                    valEncoded[idx][jdx] = vocab2Indx["<unk>"]
                jdx += 1
                if jdx == maxLen - 1:  # we have done enough encoding for now
                    break
        valEncoded[idx][jdx] = vocab2Indx["<end>"]
        idx += 1

    valDS = torch.utils.data.TensorDataset(torch.from_numpy(valEncoded))

    train_loader = torch.utils.data.DataLoader(trainDS, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valDS, shuffle=True)
    return train_loader, val_loader


def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    model = None
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss()  # is this the loss function I want to use? maybe change later
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)  # FIXME what should the lr be?

    return action_criterion, target_criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs, labels)

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(actions_out.squeeze(), labels[:, 0].long())
        target_loss = target_criterion(targets_out.squeeze(), labels[:, 1].long())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
    args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    # I'm doing two lines per graph because it would be interesting to know if we were successfully learning
    # one part (targets) but not the other (actions)

    # Loss
    trainALossTracker = list()
    trainTLossTracker = list()
    valALossTracker = list()
    valTLossTracker = list()

    # Accuracy
    trainAAccTracker = list()
    trainTAccTracker = list()
    valAAccTracker = list()
    valTAccTracker = list()


    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        trainALossTracker.append(train_action_loss)
        trainTLossTracker.append(train_target_loss)
        trainAAccTracker.append(train_action_acc)
        trainTAccTracker.append(train_target_acc)

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target losaccs: {val_target_acc}"
            )
            valALossTracker.append(val_action_loss)
            valTLossTracker.append(val_target_loss)
            valAAccTracker.append(val_action_acc)
            valTAccTracker.append(val_target_acc)

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #

    # By the time I'm out of the for loop, I have all the losses and accuracies

    trainingN = np.arange(len(trainTLossTracker))
    # graph for Training Loss
    plt.figure(1)
    plt.plot(trainingN, trainTLossTracker, label="Target")
    plt.plot(trainingN, trainALossTracker, label="Action")
    plt.legend()
    plt.title("Training Loss")

    # training accuracy
    plt.figure(2)
    plt.plot(trainingN, trainTAccTracker, label="Target")
    plt.plot(trainingN, trainAAccTracker, label="Action")
    plt.legend()
    plt.title("Training Accuracy")

    valN = np.arange(len(valTLossTracker))
    # graph for validation loss
    plt.figure(3)
    plt.plot(valN, valTLossTracker, label="Target")
    plt.plot(valN, valALossTracker, label="Action")
    plt.legend()
    plt.title("Validation Loss")

    # graph for validation accuracy
    plt.figure(4)
    plt.plot(valN, valTAccTracker, label="Target")
    plt.plot(valN, valAAccTracker, label="Action")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.show()



def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, device)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
