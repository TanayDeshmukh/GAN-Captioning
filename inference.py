import argparse
import json
import logging
import os
import re
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_transformers.modeling_bert import BertConfig
from torch.utils.data import DataLoader

from config import _C as config
from data import COCOCaptionGeneratorDataset, collate_fn_test
from model import Generator
from utils import mkdir
# from utils.checkpointer import Checkpointer
# from data import make_data_loader
from utils.logger import setup_logger
from data import EOS, MASK, tokenizer


def inference(generator, data_loader, device):
    logger = logging.getLogger("inference")
    logger.info("Start inferencing")
    generator.eval()

    pred_dict = dict()
    gt_dict = dict()

    # eos_penalizers = list()
    # for l, (low, high) in enumerate(config.boundaries):
    # pred_dict[str(l + 1)] = dict()

    # eos_penalizer = torch.ones((1, 20 - 7 + 1), dtype=torch.float, device=device)
    # eos_penalizer *= config.infer.eos_decay[l]
    # eos_penalizer = eos_penalizer.cumprod(dim=-1).flip(-1)
    # eos_penalizers.append(eos_penalizer)

    # for it, batch in enumerate(data_loader):
    #     print(it)
    #     if it==10:
    #         exit(0)

    end = time.time()
    for iteration, batch in tqdm(enumerate(data_loader), total=len(data_loader)):

        # if iteration == 6:
        #     break

        iteration = iteration + 1

        image_features = batch[0].to(device)  # (N, 100, 2048), float
        all_captions = batch[1]
        image_names = batch[2]

        # print(image_features.shape)
        # print(len(all_captions), len(all_captions[0]))
        # print(len(image_names))
        # exit(0)

        B = image_features.size(0)
        num_regions = image_features.size(1)
        
        pred_list = list()
        image_name_list = list()

        high = 20
        low = 7

        with torch.no_grad():
            batch_id = torch.arange(0, B, 1, device=device).unsqueeze(1)
           
            masked_captions = torch.full((B, 20), MASK).to(device)

            attention_mask = torch.ones((B, high + num_regions)).to(device)
            position_ids = torch.arange(20, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(masked_captions)
            
            pred_scores = generator(
                image_features,
                masked_captions,
                position_ids, attention_mask)

            pred_probs = F.softmax(pred_scores[:, num_regions:, :], dim=-1)

            pred_token_probs, pred_token_ids = pred_probs.max(dim=-1)

            total_steps = 20
            for step in range(1, total_steps):
                num_mask = max(1, int(high * (1.0 - step / total_steps)))

                mask_id = pred_token_probs.topk(num_mask, -1, False, False)[1]
                mask_id = (mask_id + batch_id * high).view(-1)

                pred_token_ids.view(-1)[mask_id] = MASK
                pred_scores = generator(
                    image_features,
                    pred_token_ids,
                    position_ids, attention_mask)

                pred_probs = F.softmax(pred_scores[:, num_regions:, :], dim=-1)
                new_token_probs, new_token_ids = pred_probs.max(dim=-1)

                pred_token_ids.view(-1)[mask_id] = new_token_ids.view(-1)[mask_id]
                pred_token_probs.view(-1)[mask_id] = new_token_probs.view(-1)[mask_id]
                pred_token_probs = (pred_token_probs + new_token_probs) / 2
                # print(tokenizer.decode(pred_token_ids[0].cpu().numpy()))


            for i in range(B):
                img_name = image_names[i]
                pred_dict[img_name] = [tokenizer.decode(pred_token_ids[i].cpu().numpy())[:-1]]
                gt_dict[img_name] = all_captions[i]

        # # print(image_ids[0])
        # for level, preds_per_level in enumerate(pred_list, 1):
        #     for batch_id, image_id in enumerate(image_ids):
        #         pred_per_level = tokenizer.decode(preds_per_level[batch_id], end_flags=[EOS])
        #         pred_per_level = re.sub(r'\b(\w+)( \1\b)+', r'\1', pred_per_level)
        #         pred_dict[str(level)][str(image_id)] = [{'caption': pred_per_level}]

    logger.info('batch_time: {time:.4f} batch_memory: {memory:.2f}'.format(
        time=(time.time() - end) / iteration,
        memory=torch.cuda.max_memory_allocated() / 1024.0 ** 3))

    return pred_dict, gt_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config.merge_from_list(args.opts)
    config.freeze()

    save_dir = os.path.join(config.save_dir)
    # mkdir(save_dir)
    logger = setup_logger("inference", save_dir, 0)
    logger.info("Running with config:\n{}".format(config))

    device = torch.device(config.device)
    num_types =  2

    generator = Generator(BertConfig())

    checkpoint = torch.load(os.path.join(config.model_path, 'best_model.pth'))
    generator.load_state_dict(checkpoint['state_dict'])

    generator = generator.to(device)
    # g_checkpointer = Checkpointer(model=generator, logger=logger)
    # g_checkpointer.load(config.model_path, True)

    dataset = COCOCaptionGeneratorDataset(
        root=config.root_dir,
        split='test'

    )
    # data_loader = make_data_loader(
    #     dataset=dataset,
    #     batch_size=8,
    #     num_workers=0,
    #     split='test'
    # )

    data_loader = DataLoader(
        dataset,
        batch_size = 8,
        shuffle = False,
        num_workers = 6,
        drop_last = True,
        collate_fn=collate_fn_test
    )

    # print('---------------------------', len(data_loader))
    # exit(0)

    pred_dict, gt_dict = inference(generator, data_loader, device)
    logger.info(f"Saving results to {save_dir}/caption_results.json")
    with open(os.path.join(save_dir, 'caption_results.json'), 'w') as f:
        json.dump(pred_dict, f)
    with open(os.path.join(save_dir, 'gt_captions.json'), 'w') as f:
        json.dump(gt_dict, f)
