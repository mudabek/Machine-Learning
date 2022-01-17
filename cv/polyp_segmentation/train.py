import sys
import argparse
import yaml
import pathlib

import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

# sys.path.append('../src/')
import dataset
import transforms
import losses
import metrics
import trainer
import models
import cvcclinicdb

from smallunet import UNetSmall
from CCBANet import CCBANetModel

def main(args):
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    # read config:
    path_to_data = pathlib.Path(config['path_to_data'])
    path_to_save_dir = pathlib.Path(config['path_to_save_dir'])

    train_batch_size = int(config['train_batch_size'])
    val_batch_size = int(config['val_batch_size'])
    num_workers = int(config['num_workers'])
    lr = float(config['lr'])
    n_epochs = int(config['n_epochs'])

    # datasets:
    train_set = cvcclinicdb.CVCClinicDBDataset(mode="train",normalization=True,augmentation=True)#dataset.PolypDataset(root=path_to_data, mode='train', transforms=train_transforms)
    val_set = cvcclinicdb.CVCClinicDBDataset(mode="test",normalization=True,augmentation=True)#dataset.PolypDataset(root=path_to_data, mode='test', transforms=val_transforms)

    # dataloaders:
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # model = models.UNet(input_channel=3, output_channel=1)
    model = CCBANetModel(n_classes=1)

    criterion = losses.CombinedLoss()#losses.TotalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-5)
    metric = metrics.dice
    metric_2 = metrics.jaccard

    trainer_ = trainer.ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        metric=metric,
        metric_2=metric_2,
        num_epochs=n_epochs,
        parallel=True,
    )

    trainer_.train_model()
    trainer_.save_results(path_to_dir=path_to_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)