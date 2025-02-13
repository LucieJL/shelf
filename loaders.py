import os
from glob import glob

from torch.utils.data import DataLoader, random_split
import lightning as pl
from lightning.pytorch.loggers import CSVLogger

from perch_dataset import PerchDataset
from perch_mlp import PerchMLP

def load_data(cfg, test_set, infer=False):
    print('Loading data... ')
    train_val_annotation_file = os.path.join('annotations', cfg.pathing.subsamp_annotation_file)
    test_annotation_file = os.path.join('annotations', test_set, cfg.pathing.full_annotation_file)

    train_val_set = PerchDataset(annotation_file=train_val_annotation_file,
                                 data_path=cfg.pathing.data_path,
                                 encoder_path=cfg.pathing.encoder_path,
                                 params=cfg.load_data_params,
                                 test_set=test_set, test_mode=False)
    train_set, val_set = random_split(train_val_set, cfg.train_params.train_val_ratio)

    test_set = PerchDataset(annotation_file=test_annotation_file,
                            data_path=cfg.pathing.data_path,
                            encoder_path=cfg.pathing.encoder_path,
                            params=cfg.load_data_params,
                            test_set=test_set, test_mode=True)
    print(f'test : {len(test_set)}')

    train_loader = DataLoader(train_set, cfg.train_params.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, cfg.train_params.batch_size)
    test_loader = DataLoader(test_set, cfg.train_params.batch_size)

    return train_loader, val_loader, test_loader


def load_model(cfg, log_path, machine, load_from, test_set):
    print('Loading model... ')
    checkpoint_path = os.path.join(machine, 'logs', f'perch_raw{cfg.audio_params.dur}',
                                   load_from, test_set, 'checkpoints')

    if os.path.exists(checkpoint_path):
        print('Checkpoint found. Loading model from checkpoint...')
        checkpoint_file = glob(os.path.join(checkpoint_path, '*.ckpt')) # returns a list
        return PerchMLP.load_from_checkpoint(checkpoint_file[0],
                                             n_in=cfg.perch_params.output_size,
                                             n_out=len(cfg.train_params.labels),
                                             lr=cfg.train_params.lr,
                                             log_path=log_path
                                             )


    else:
        print('No checkpoint found. Initializing new model...')
        return PerchMLP(n_in=cfg.perch_params.output_size,
                    n_out=len(cfg.train_params.labels),
                    lr=cfg.train_params.lr,
                    log_path=log_path)


def load_trainer(cfg, machine, now, test_set):
    print('Loading trainer... ')

    logger = CSVLogger(save_dir=os.path.join(machine, 'logs'),
                       name=cfg.xp_name,
                       version=os.path.join(now, test_set))

    trainer = pl.Trainer(
        max_epochs=cfg.pl_trainer_params.n_epochs,
        logger=logger
    )

    return trainer
