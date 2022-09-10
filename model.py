import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# based on a pytorch tutorial
class commandIDer(nn.Module):
    def __init__(self, embedDim, hidDim, vocabSize, actDim, targetDim):
        super(commandIDer, self).__init__()
        self.hidDim = hidDim

        self.embeddings = nn.Embedding(vocabSize, embedDim)

        self.lstm = nn.LSTM(embedDim, hidDim)

        # I'm honestly not sure if these are learning together or separate, but
        # I assume it's together because it's in the same model
        # Mapping from hidden layer to the action output
        self.toAct = nn.Linear(hidDim, actDim)
        # From hidden layer to target output
        self.toTarget = nn.Linear(hidDim, targetDim)

    def forward(self, command):
        embeds = self.embeddings(command)  # command should already be a list of embedded words
        # print(embeds.shape)  # 1, 24, 128
        # print(len(command))  # maybe not expecting doble [[]]? so do of 0?
        # print(len(command[0]))
        # print(command)
        # print((embeds.view(len(command[0]), 1, -1)).shape)

        lstmOut, _ = self.lstm(embeds.view(len(command[0]), 1, -1))
        actSpace = self.toAct(lstmOut.view(len(command[0]), -1))
        targetSpace = self.toTarget(lstmOut.view(len(command[0]), -1))
        actScores = F.log_softmax(actSpace, dim=1)
        targetScores = F.log_softmax(targetSpace, dim=1)

        print(actScores)

        return actScores, targetScores

