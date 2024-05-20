"""Training module"""
from pathlib import Path
import argparse
import os

import lightning.pytorch as pl
from lightning.pytorch.loggers import NeptuneLogger, TensorBoardLogger

from datamodules.datamodule import DataModule
from models.model import Model
from utils.helpers import load_config


def train(args):
    """
    :param args: parsed arguments
    :rtype: None
    """
    # Overrides used graphic card
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = load_config(args.config)
    token = config['config']['NEPTUNE_API_TOKEN']
    project = config['config']['NEPTUNE_PROJECT']

    if args.use_neptune:
        logger = NeptuneLogger(
            project=project,
            api_token=token)
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    pl.seed_everything(42, workers=True)
    patience = 25

    datamodule = DataModule(
        data_path=Path('data'),
        batch_size=16,
        num_workers=4,
        train_size=0.8
    )

    model = Model(
        lr=2.55e-5,
        lr_patience=5,
        lr_factor=0.5,
        n_classes=1000,
        model_selection="resnet18"
    )

    model.hparams.update(datamodule.hparams)

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=-1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch}-{val_MeanAbsoluteError:.5f}', mode='min',
                                                       monitor='val_MeanAbsoluteError', verbose=True, save_last=True)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_MeanAbsoluteError', mode='min', patience=patience)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        logger=logger,
        devices=1,
        callbacks=[model_summary_callback, checkpoint_callback, early_stop_callback, lr_monitor],
        accelerator='cuda',  # change to 'cpu' if needed
        max_epochs=args.epochs
    )

    trainer.fit(model=model, train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader())

    results = trainer.test(model=model, ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-c', '--config', action='store', default='../config.yaml')
    parser.add_argument('-e', '--epochs', action='store', default=50,
                        type=int, help='Specified number of maximum epochs')
    parser.add_argument('-d', '--data', action='store', default="../data",
                        type=str, help='Path to root folder of data')
    parser.add_argument('-n', '--use-neptune', action='store', type=bool, default=False,
                        help="Use neptune logger with credentials provided in config")
    args_parsed = parser.parse_args()
    train(args_parsed)
