import torch
import time
import logging

import torch.nn.functional as F

from config import _C as config
from model import LabelSmoothingLoss
from torch.nn.utils import clip_grad_norm_
from data import EOS, MASK, PAD, tokenizer, num_tokens


def train(generator, data_loader, device, optimizer, scheduler,
        criterion, log_time, epoch):
    
    print('traning...')
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train")
    logger.info("Start training")

    max_iter = len(data_loader)
    generator.train()

    running_loss = 0.0

    end = time.time()
    for iteration, batch in enumerate(data_loader):

        # if iteration >= 20:
        #     break
        captions = batch[0].to(device)  # (N, L), long
        masked_captions = batch[1].to(device)  # (N, L), long
        image_features = batch[2].to(device)  # (N, L), long

        num_img_tokens = image_features.size(1)
        seq_length = captions.size(1)
        batch_size = captions.size(0)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(captions)

        attention_mask = (masked_captions != PAD).float()
        _attention_mask = attention_mask.new_ones((batch_size, num_img_tokens))
        attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

        mask_position = (masked_captions == MASK).to(torch.long).view(-1)
        mask_position = mask_position.nonzero().squeeze()

        pred_scores = generator(
            image_features,
            masked_captions,
            position_ids, attention_mask)

        pred_scores = pred_scores[:, num_img_tokens:, :]
        pred_scores = pred_scores.contiguous().view(-1, num_tokens)
        pred_scores = pred_scores[mask_position]

        gt_token_ids = captions.view(-1)[mask_position]

        loss = criterion(pred_scores, gt_token_ids)

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(generator.parameters(), config.solver.grad_clip)
        optimizer.step()
        scheduler.step()
        batch_time = time.time() - end
        end = time.time()

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


def val(generator, data_loader, device, criterion,
        log_time, epoch):

    # print("Validating...")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("validation")
    logger.info("Start Validation")

    max_iter = len(data_loader)
    generator.train()

    running_loss = 0.0

    end = time.time()
    for iteration, batch in enumerate(data_loader):

        # if iteration >=20:
        #     break

        captions = batch[0].to(device)  # (N, L), long
        masked_captions = batch[1].to(device)  # (N, L), long
        image_features = batch[2].to(device)  # (N, L), long

        num_img_tokens = image_features.size(1)
        seq_length = captions.size(1)
        batch_size = captions.size(0)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(captions)

        attention_mask = (masked_captions != PAD).float()
        _attention_mask = attention_mask.new_ones((batch_size, num_img_tokens))
        attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

        mask_position = (masked_captions == MASK).to(torch.long).view(-1)
        mask_position = mask_position.nonzero().squeeze()

        pred_scores = generator(
            image_features,
            masked_captions,
            position_ids, attention_mask)

        pred_scores = pred_scores[:, num_img_tokens:, :]
        pred_scores = pred_scores.contiguous().view(-1, num_tokens)
        pred_scores = pred_scores[mask_position]

        gt_token_ids = captions.view(-1)[mask_position]

        loss = criterion(pred_scores, gt_token_ids)

        running_loss += loss.item()

        batch_time = time.time() - end
        end = time.time()

        if iteration % log_time == 0 or iteration == max_iter:
            logger.info(
                '  '.join([
                    "epoch: {epoch}", "iter: {iter}", "time: {time:.4f}", 
                    "mem: {mem:.2f}", "loss: {loss:.4f}"
                ]).format(
                    epoch=epoch, iter=iteration, time=batch_time, loss=loss,
                    mem=torch.cuda.max_memory_allocated() / 1024.0 ** 3,
                ))
       
    return running_loss / len(data_loader)
    # return running_loss / 20

