import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from preprocess import *


class convNet(nn.Module):
    """
    copies the neural net used in a paper.
    "Improved musical onset detection with Convolutional Neural Networks".
    src: https://ieeexplore.ieee.org/document/6854953
    """

    def __init__(self):

        super(convNet, self).__init__()
        # model
        self.conv1 = nn.Conv2d(3, 10, (3, 7))
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(1120, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 1)

    def forward(self, x, istraining=False, minibatch=1):

        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 1))
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 1))
        x = F.dropout(x.view(minibatch, -1), training=istraining)
        x = F.dropout(F.relu(self.fc1(x)), training=istraining)
        x = F.dropout(F.relu(self.fc2(x)), training=istraining)

        return F.sigmoid(self.fc3(x))

    def infer_data_builder(self, feats, soundlen=15, minibatch=1):

        x = []

        for i in range(feats.shape[2] - soundlen):
            x.append(feats[:, :, i : i + soundlen])

            if (i + 1) % minibatch == 0:
                yield (torch.from_numpy(np.array(x)).float())
                x = []

        if len(x) != 0:
            yield (torch.from_numpy(np.array(x)).float())

    def infer(self, feats, device, minibatch=1):

        with torch.no_grad():
            inference = None
            for x in tqdm(
                self.infer_data_builder(feats, minibatch=minibatch),
                total=feats.shape[2] // minibatch,
            ):
                output = self(x.to(device), minibatch=x.shape[0])
                if inference is not None:
                    inference = np.concatenate(
                        (inference, output.cpu().numpy().reshape(-1))
                    )
                else:
                    inference = output.cpu().numpy().reshape(-1)

            return np.array(inference).reshape(-1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = convNet()
    net = net.to(device)

    print(net)
    print("parameters: ", sum(p.numel() for p in net.parameters()))
