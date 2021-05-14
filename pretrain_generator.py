import os
import torch
import numpy as np
from shutil import copyfile
from trainer import train, val
from config import _C as config
from torch.utils.data import DataLoader
from utils import setup_logger, get_rank
from model import Generator, LabelSmoothingLoss
from torch.utils.tensorboard import SummaryWriter
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers.optimization import AdamW, WarmupCosineSchedule
from data import COCOCaptionGeneratorDataset, collate_fn_train, num_tokens, EOS

if __name__ =='__main__':

    device = torch.device(config.device)

    # logger = setup_logger("pretrain_generator", config.save_dir, get_rank())
    # logger.info("Running with config:\n{}".format(config))

    writer = SummaryWriter(log_dir='./tensorboard_logs')

    bert_config = BertConfig()
    generator = Generator(bert_config).to(device)

    patience = 0
    min_loss = np.inf

    print("Generating Dataset...")

    train_dataset = COCOCaptionGeneratorDataset(
        root = config.root_dir,
        split = 'train'
    )

    val_dataset = COCOCaptionGeneratorDataset(
        root = config.root_dir,
        split = 'val'
    )

    print("Creating Dataloader...")

    train_data_loader = DataLoader(
        train_dataset,
        batch_size = config.train_batch_size,
        shuffle = True,
        num_workers = 12,
        drop_last = True,
        collate_fn = collate_fn_train
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size = config.val_batch_size,
        shuffle = False,
        num_workers = 12,
        drop_last = True,
        collate_fn = collate_fn_train
    )

    if config.loss.balance_weight != 1.0:
        balance_weight = torch.ones(
            num_tokens, dtype=torch.float32, device=device)
        balance_weight[EOS] = config.loss.balance_weight
    else:
        balance_weight = None

    criterion = LabelSmoothingLoss(
        num_tokens, balance_weight, config.loss.label_smoothing)

    optimizer = AdamW(
        params=generator.parameters(),
        lr=config.solver.lr,
        weight_decay=config.solver.weight_decay,
        betas=config.solver.betas
    )

    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=config.scheduler.warmup_steps,
        t_total=config.scheduler.max_steps
    )

    print('Start training...')

    for epoch in range(config.num_epochs):

        train_loss = train(
            generator=generator,
            data_loader=train_data_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            log_time=config.log_time,
            epoch=epoch)
        
        # print("training loss : ", train_loss)

        val_loss = val(
            generator=generator,
            data_loader=val_data_loader,
            device=device,
            criterion=criterion,
            log_time=config.log_time,
            epoch=epoch)
        
        # print("valdation loss : ", val_loss)

        writer.add_scalar('Training loss', train_loss, epoch)
        writer.add_scalar('Validation loss', val_loss, epoch)

        best = False
        if val_loss <= min_loss:
            min_loss = val_loss
            patience = 0
            best = True
        else:
            patience += 1

        if patience > 5:
            print("Patience reached...")
            break
        
        print("Saving model...")

        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'state_dict': generator.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
        }, os.path.join(config.model_path, 'last_model.pth'))

        if best:
            copyfile(os.path.join(config.model_path, 'last_model.pth') , os.path.join(config.model_path, 'best_model.pth'))
