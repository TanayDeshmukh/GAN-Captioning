import os
import torch
import time
import random
import logging
import numpy as np
from os import EX_IOERR

# from model import Discriminator, DistilBertModel
from torch import nn
from shutil import copyfile
from model import Discriminator
from config import _C as config
from torch.utils.data import DataLoader
from utils import setup_logger, get_rank
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from data import PAD
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers.optimization import AdamW, WarmupCosineSchedule
from data import COCOCaptionDiscriminatorDataset, collate_fn_discriminator_train


def train(discriminator, data_loader, device, optimizer, scheduler,
        criterion, log_time, epoch):
    
    print('traning...')
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train")
    logger.info("Start training")

    max_iter = len(data_loader)
    discriminator.train()

    running_loss = 0.0

    end = time.time()
    for iteration, (captions, image_features, image_name, label) in enumerate(train_data_loader):
        
        captions = captions.to(device)  # (N, L), long
        image_features = image_features.to(device)  # (N, L), long

        num_img_tokens = image_features.size(1)
        seq_length = captions.size(1)
        batch_size = captions.size(0)
        label = label.to(device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(captions)

        attention_mask = (captions != PAD).float()
        _attention_mask = attention_mask.new_ones((batch_size, num_img_tokens))
        attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

        pred_scores = discriminator(
                captions,
                position_ids,
                image_features,
                attention_mask)
        
        loss = criterion(pred_scores, label)

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(discriminator.parameters(), config.solver.grad_clip)

        optimizer.step()
        scheduler.step()
        batch_time = time.time() - end
        end = time.time()

        # print("train loss : ", loss)

        if iteration % 10 == 0:
            break

        if iteration % log_time == 0 or iteration == max_iter:
            logger.info(
                '  '.join([
                    "epoch: {epoch}", "iter: {iter}", "time: {time:.4f}", 
                    "mem: {mem:.2f}", "lr: {lr:.8f}", "loss: {loss:.4f}"
                ]).format(
                    epoch=epoch, iter=iteration, time=batch_time, loss=loss,
                    lr=optimizer.param_groups[0]["lr"],
                    mem=torch.cuda.max_memory_allocated() / 1024.0 ** 3,
                ))
       
    return running_loss / len(data_loader)
    # return running_loss / 20


def val(discriminator, data_loader, device, criterion,
        log_time, epoch):

    print('Validating...')
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("val")
    logger.info("Start validation")

    max_iter = len(data_loader)
    discriminator.eval()

    running_loss = 0.0

    end = time.time()
    for iteration, (captions, image_features, image_name, label) in enumerate(train_data_loader):
        
        captions = captions.to(device)  # (N, L), long
        image_features = image_features.to(device)  # (N, L), long

        num_img_tokens = image_features.size(1)
        seq_length = captions.size(1)
        batch_size = captions.size(0)
        label = label.to(device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(captions)

        attention_mask = (captions != PAD).float()
        _attention_mask = attention_mask.new_ones((batch_size, num_img_tokens))
        attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

        pred_scores = discriminator(
                captions,
                position_ids,
                image_features,
                attention_mask)
        
        loss = criterion(pred_scores, label)

        running_loss += loss.item()

        batch_time = time.time() - end
        end = time.time()

        # print("val loss : ", loss)


        if iteration % log_time == 0 or iteration == max_iter:
            logger.info(
                '  '.join([
                    "epoch: {epoch}", "iter: {iter}", "time: {time:.4f}", 
                    "mem: {mem:.2f}", "lr: {lr:.8f}", "loss: {loss:.4f}"
                ]).format(
                    epoch=epoch, iter=iteration, time=batch_time, loss=loss,
                    lr=optimizer.param_groups[0]["lr"],
                    mem=torch.cuda.max_memory_allocated() / 1024.0 ** 3,
                ))
       
    return running_loss / len(data_loader)
    # return running_loss / 20



if __name__ == '__main__':

    device = torch.device(config.device)
    # device = "cpu"

    writer = SummaryWriter(log_dir='./tensorboard_logs')

    bert_config = BertConfig()
    discriminator = Discriminator(bert_config)
    discriminator = discriminator.to(device)

    # print(discriminator)
    # exit(0)

    patience = 0
    min_loss = np.inf

    train_dataset = COCOCaptionDiscriminatorDataset(
            root = config.root_dir,
            split = 'train'
        )

    train_data_loader = DataLoader(
            train_dataset,
            batch_size = config.train_batch_size,
            shuffle = True,
            num_workers = 0,
            drop_last = True,
            collate_fn = collate_fn_discriminator_train
        )
    
    val_dataset = COCOCaptionDiscriminatorDataset(
            root = config.root_dir,
            split = 'val'
        )

    val_data_loader = DataLoader(
            val_dataset,
            batch_size = config.train_batch_size,
            shuffle = True,
            num_workers = 0,
            drop_last = True,
            collate_fn = collate_fn_discriminator_train
        )

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        params=discriminator.parameters(),
        lr=config.solver.lr,
        weight_decay=config.solver.weight_decay,
        betas=config.solver.betas
    )

    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=config.scheduler.warmup_steps,
        t_total=config.scheduler.max_steps
    )

    for epoch in range(config.num_epochs):

        train_loss = train(
            discriminator=discriminator,
            data_loader=train_data_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            log_time=config.log_time,
            epoch=epoch)
        
        # print("training loss : ", train_loss)

        val_loss = val(
            discriminator=discriminator,
            data_loader=val_data_loader,
            device=device,
            criterion=criterion,
            log_time=config.log_time,
            epoch=epoch)
        
        # print("valdation loss : ", val_loss)

        writer.add_scalar('discriminator/Training loss', train_loss, epoch)
        writer.add_scalar('discriminator/Validation loss', val_loss, epoch)

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
            'state_dict': discriminator.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
        }, os.path.join(config.model_path, 'last_model_discriminator.pth'))

        if best:
            copyfile(os.path.join(config.model_path, 'last_model_discriminator.pth') , os.path.join(config.model_path, 'best_model_discriminator.pth'))


    # for idx, (captions, image_features, image_name, label) in enumerate(train_data_loader):
        
    #     captions = captions.to(device)  # (N, L), long
    #     image_features = image_features.to(device)  # (N, L), long

    #     num_img_tokens = image_features.size(1)
    #     seq_length = captions.size(1)
    #     batch_size = captions.size(0)

    #     position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    #     position_ids = position_ids.unsqueeze(0).expand_as(captions)

    #     attention_mask = (captions != PAD).float()
    #     _attention_mask = attention_mask.new_ones((batch_size, num_img_tokens))
    #     attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

    #     pred_scores = discriminator(
    #             captions,
    #             position_ids,
    #             image_features,
    #             attention_mask)
        
    #     # print(pred_scores)
    #     # print(label)

    #     loss = lossfn(pred_scores, label)
    #     print(loss)
            
    #     exit(0)