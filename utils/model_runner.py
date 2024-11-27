import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch

from utils.model_selector import select_enc_model, select_model
from nets.generic_classifier import GenericClassifier
import os

class ConditionalEarlyStopping(EarlyStopping):
    def __init__(self, train_metric_threshold=0.1, **kwargs):
        super().__init__(**kwargs)
        self.train_metric_threshold = train_metric_threshold
        self.stop_monitoring = True 

    def on_train_epoch_end(self, trainer, pl_module):
        train_eer = trainer.callback_metrics.get("train_eer")

        if train_eer is not None and train_eer < self.train_metric_threshold:
            self.stop_monitoring = False 

        if not self.stop_monitoring:
            super().on_train_epoch_end(trainer, pl_module)

def start_enc_training(args, train_loader, dev_loader):
    encoder_model = select_enc_model(args)
    model_filename = (
        f"encoder_{args.model}_{args.dataset}_ep{args.epoch}_bs{args.batch_size}"
    )

    encoder_checkpoint_callback = ModelCheckpoint(
        monitor="val_alignment_loss",
        mode="min",
        dirpath="models/",
        filename=model_filename,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_alignment_loss",
        min_delta=0.00007,
        patience=3,
        verbose=True,
        mode="min",
    )

    torch.set_float32_matmul_precision("medium")
    encoder_trainer = pl.Trainer(
        precision="bf16-mixed",
        logger=False,
        max_epochs=args.epoch,
        # limit_train_batches=0.02,  # For debugging purposes: simulate pre-training using a small subset of training data
        # limit_val_batches=0.02,    # For debugging purposes: simulate pre-training using a small subset of validation data
        accelerator=args.accelerator,
        devices=1,
        callbacks=[encoder_checkpoint_callback, early_stop_callback],
    )

    encoder_trainer.fit(
        encoder_model, train_dataloaders=train_loader, val_dataloaders=dev_loader
    )

    print("Saved model:", encoder_checkpoint_callback.best_model_path)
    return os.path.basename(encoder_checkpoint_callback.best_model_path)

def start_cls_training(args, train_loader, dev_loader, eval_loader):
    if args.encoder_file is not None:
        print(args.encoder_file)
        ckpt = torch.load(f"./models/{args.encoder_file}")
        cls_model = select_model(args)
        cls_model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        cls_model = select_model(args)

    classifier = GenericClassifier(cls_model, args)
    filename_prefix = "pretrained" if args.encoder_file is not None else "scratch"
    common_name_part = f"{filename_prefix}_{args.model}_{args.dataset}_ep{args.epoch}_bs{args.batch_size}_lb{int(args.label_ratio*100)}"

    cls_checkpoint_callback = ModelCheckpoint(
        monitor="valid_eer", mode="min", dirpath="models/", filename=common_name_part
    )

    early_stop_callback = ConditionalEarlyStopping(
        train_metric_threshold=0.1,
        monitor="valid_eer",
        min_delta=0.00, 
        # patience=5, 
        patience=3, 
        verbose=True, 
        mode="min"
    )
    
    print("number of epoch:", args.epoch)
    torch.set_float32_matmul_precision("high")
    cls_trainer = pl.Trainer(
        precision="bf16-mixed",
        logger=False,
        max_epochs=args.epoch,
        accelerator=args.accelerator,
        devices=1,
        callbacks=[cls_checkpoint_callback, early_stop_callback],
    )

    cls_trainer.fit(
        classifier, train_dataloaders=train_loader, val_dataloaders=dev_loader
    )

    cls_trainer.test(dataloaders=eval_loader, ckpt_path="best")

def cls_eval_only(args, eval_loader):
    cls_model = select_model(args)
    classifier = GenericClassifier(cls_model, args)
    ckpt = torch.load(f"./models/{args.encoder_file}")
    classifier.load_state_dict(ckpt["state_dict"], strict=True)

    torch.set_float32_matmul_precision("high")
    cls_trainer = pl.Trainer(args.accelerator, devices=1, logger=False, enable_progress_bar=True)
    cls_trainer.test(model=classifier, dataloaders=eval_loader)
