import argparse
import os
import random
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset

from dataloader.gafa import create_gafa_dataset
from models.gazenet import GazeNet, SimpleRHFDGazeNet


def train(opt):
    if opt.simple:
        model = SimpleRHFDGazeNet(n_frames=opt.n_frames)
        # Load pretrained HBNet weights
        if opt.weights and os.path.exists(opt.weights):
            model.load_pretrained_hbnet(opt.weights)
            print(f"Loaded HBNet weights from {opt.weights}")
        else:
            print("WARNING: No pretrained weights found, HBNet randomly initialized")
    else:
        model = GazeNet(
            n_frames=opt.n_frames,
            # RHFD features
            use_rhfd_features=opt.use_rhfd_features,
            use_probability_head=opt.use_probability_head,
            # TWIESN
            use_twiesn=opt.use_twiesn,
            twiesn_reservoir_dim=opt.twiesn_reservoir_dim,
            twiesn_tchebichef_order=opt.twiesn_tchebichef_order,
            twiesn_spectral_radius=opt.twiesn_spectral_radius,
            # EMA
            use_ema=opt.use_ema,
            ema_alpha=opt.ema_alpha,
            use_adaptive_ema=opt.use_adaptive_ema,
            # Probability head
            n_anchors=opt.n_anchors,
        )

    # default training dataset
    if opt.quick:
        train_exp_names = ['living_room/004']
    else:
        train_exp_names = [
            'library/1026_3',
            'library/1028_2',
            'library/1028_5',
            'lab/1013_1',
            'lab/1014_1',
            'kitchen/1022_4',
            'kitchen/1015_4',
            'living_room/004',
            'living_room/005',
            'courtyard/004',
            'courtyard/005',
        ]

    random.shuffle(train_exp_names)
    dset = create_gafa_dataset(
        n_frames=opt.n_frames, exp_names=train_exp_names, interval=1
    )
    train_idx = np.arange(0, int(len(dset) * 0.9))
    val_idx = np.arange(int(len(dset) * 0.9), len(dset))
    train_dset = Subset(dset, train_idx)
    validation_dset = Subset(dset, val_idx)

    checkpoint_collback = ModelCheckpoint(monitor="val_loss", save_top_k=1)

    trainer = Trainer(
        default_root_dir=opt.checkpoint,
        callbacks=[checkpoint_collback],
        benchmark=True,
        min_epochs=opt.epoch,
        max_epochs=opt.epoch,
        gpus=opt.gpus,
        strategy="ddp",
        precision=16,
    )

    train_loader = DataLoader(
        train_dset, batch_size=opt.batch_size, num_workers=4, pin_memory=True, shuffle=True
    )
    val_loader = DataLoader(
        validation_dset, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--n_frames", type=int, default=7)
    parser.add_argument("--checkpoint", type=str, default="output/")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--quick", action="store_true", default=False,
        help="Quick debug mode: single scene + fewer epochs"
    )
    parser.add_argument(
        "--simple", action="store_true", default=False,
        help="Use SimpleRHFDGazeNet (minimal stable enhancement: only Gf+Gd channels)"
    )
    parser.add_argument(
        "--weights", type=str, default="./models/weights/gazenet_GAFA.pth",
        help="Path to pretrained GAFA checkpoint (for HBNet weights)"
    )

    # RHFD features
    parser.add_argument(
        "--use_rhfd_features", action="store_true", default=True,
        help="Enable Gf/Gd RHFD feature extraction"
    )
    parser.add_argument(
        "--no_rhfd_features", action="store_false", dest="use_rhfd_features",
        help="Disable RHFD features"
    )
    parser.add_argument(
        "--use_probability_head", action="store_true", default=True,
        help="Use probability-first gaze mapping"
    )
    parser.add_argument(
        "--no_probability_head", action="store_false", dest="use_probability_head",
        help="Use direct regression instead of probability head"
    )

    # TWIESN
    parser.add_argument(
        "--use_twiesn", action="store_true", default=True,
        help="Enable TWIESN temporal smoothing"
    )
    parser.add_argument(
        "--no_twiesn", action="store_false", dest="use_twiesn",
        help="Disable TWIESN"
    )
    parser.add_argument("--twiesn_reservoir_dim", type=int, default=128)
    parser.add_argument("--twiesn_tchebichef_order", type=int, default=4)
    parser.add_argument("--twiesn_spectral_radius", type=float, default=0.9)

    # Exponential smoothing
    parser.add_argument(
        "--use_ema", action="store_true", default=True,
        help="Enable exponential smoothing"
    )
    parser.add_argument(
        "--no_ema", action="store_false", dest="use_ema",
        help="Disable EMA"
    )
    parser.add_argument("--ema_alpha", type=float, default=0.3)
    parser.add_argument(
        "--use_adaptive_ema", action="store_true", default=False,
        help="Use kappa-adaptive EMA"
    )

    # Probability head
    parser.add_argument("--n_anchors", type=int, default=64)

    opt = parser.parse_args()

    train(opt)
