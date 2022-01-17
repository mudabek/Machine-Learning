import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torchvision.transforms as transforms 
import torchvision
from torch.utils.data import DataLoader
import sys

from sklearn import metrics
import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

from custom_transforms import RandomNoiseDenoise, Bilateral, ToRGB
from dataset import BreastCancerDataset
import torch.nn.functional  as F
from torch import nn


train_dataset = BreastCancerDataset('train_data.csv', transform=transforms.Compose([
                              RandomNoiseDenoise(),
                              transforms.ToTensor(),
                              transforms.ToPILImage(),
                              transforms.Resize((224,224)),
                              transforms.RandomVerticalFlip(),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomRotation(15),
                              transforms.ToTensor(),
                              ToRGB(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                              ]))
                              

test_dataset = BreastCancerDataset('test_data.csv', transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.ToPILImage(),
                              transforms.Resize((224,224)),
                              transforms.ToTensor(),
                              ToRGB(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                              ]))


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# [Experiment 2]
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def train_and_evaluate(epochs, optimizer, model, criterion, path_to_dir):
    """ Function for training the model [taken from lab activity] """
    best_accuracy = 0
    learning_curves = dict()
    learning_curves['loss'], learning_curves['metric'] = dict(), dict()
    learning_curves['loss']['train'], learning_curves['loss']['test'] = [], []
    learning_curves['metric']['train'], learning_curves['metric']['test'] = [], []


    for epoch_num in range(0, epochs):
        model.train()
        epoch_total_loss = 0
        labels = []
        predictions = []

        # Train 
        for sample in tqdm(train_loader):
            # Forward pass
            inp, target = sample['image'], sample['label_id']
            optimizer.zero_grad()
            inp = inp.to(device)
            output = model(inp)
            labels+=target
            _, batch_prediction = torch.max(output, dim=1)
            predictions += batch_prediction.detach().tolist()

            # L2 Regularization [Experiment 3]
            l2_reg = torch.autograd.Variable( torch.FloatTensor(1), requires_grad=True).to(device)
            for W in model.parameters():
                l2_reg = l2_reg + W.norm(2)
                
            l2_reg = torch.clip(l2_reg, 0, 1.5)
            if torch.isnan(l2_reg) == True:
                l2_reg = torch.tensor([0]).to(device)

            # Backward pass
            target = target.to(device)
            batch_loss = criterion(output,target) + l2_reg * 0.001
            epoch_total_loss += batch_loss.item()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        accuracy = metrics.accuracy_score(labels, predictions)
        avrg_loss = epoch_total_loss / train_dataset.__len__()

        learning_curves['loss']['train'].append(avrg_loss)
        learning_curves['metric']['train'].append(accuracy)

        print("Train Accuracy = %0.4f" % (accuracy))
        print("Epoch %d - loss=%0.4f" % (epoch_num, avrg_loss))
        
        # Evaluate
        epoch_total_loss = 0
        with torch.no_grad():
            model.eval()
            labels = []
            predictions = []
            for sample in test_loader:
                inp, target = sample['image'], sample['label_id']
                inp = inp.to(device)
                labels+=target
                output = model(inp)
                _, batch_prediction = torch.max(output, dim=1)
                predictions += batch_prediction.detach().tolist()

                batch_loss = criterion(output, target.to(device))
                epoch_total_loss += batch_loss.item()

            accuracy = metrics.accuracy_score(labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), path_to_dir + '/model_weights.pth')

            avrg_loss = epoch_total_loss / test_dataset.__len__()

            learning_curves['loss']['test'].append(avrg_loss)
            learning_curves['metric']['test'].append(accuracy)

            print("Test Accuracy = %0.4f" % (accuracy))
            confusion = metrics.confusion_matrix(labels, predictions)
            print(confusion)

    # Save learning curves as pandas df:
    df_learning_curves = pd.DataFrame.from_dict({
        'loss_train': learning_curves['loss']['train'],
        'loss_val': learning_curves['loss']['test'],
        'metric_train': learning_curves['metric']['train'],
        'metric_val': learning_curves['metric']['test']
    })
    df_learning_curves.to_csv(path_to_dir + '/learning_curves.csv', sep=';')

    # Save learning curves' plots in png files:
    # Loss figure:
    plt.figure(figsize=(17.5, 10))
    plt.plot(range(epochs), learning_curves['loss']['train'], label='train')
    plt.plot(range(epochs), learning_curves['loss']['test'], label='test')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.grid()
    plt.savefig(path_to_dir + '/loss_plot.png', bbox_inches='tight')

    # metric figure:
    train_avg_metric = [np.mean(i) for i in learning_curves['metric']['train']]
    val_avg_metric = [np.mean(i) for i in learning_curves['metric']['test']]

    plt.figure(figsize=(17.5, 10))
    plt.plot(range(epochs), train_avg_metric, label='train')
    plt.plot(range(epochs), val_avg_metric, label='test')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Avg metric', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.grid()
    plt.savefig(path_to_dir + '/metric_plot.png', bbox_inches='tight')

    print(f'All results have been saved in {path_to_dir}')
    



model = torchvision.models.densenet121(pretrained=True)
model = model.to(device)
model.classifier = torch.nn.Linear(1024, 3).to(device)



epochs = 30
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
class_weights = torch.Tensor()
criterion = FocalLoss()


train_and_evaluate(epochs, optimizer, model, criterion, 'experiment_5')