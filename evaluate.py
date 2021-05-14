import os
import json
import argparse

import numpy as np
from config import _C as config
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--gt_caption", type=str)
    parser.add_argument("--pd_caption", type=str)
    # parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    gt_caption = os.path.join(config.save_dir, 'gt_captions.json')
    pd_caption = os.path.join(config.save_dir, 'caption_results.json')

    logger = setup_logger("evaluate", config.save_dir, 0)
    ptb_tokenizer = PTBTokenizer()

    # scorers = [(Cider(), "C"), (Spice(), "S"),
    #            (Bleu(4), ["B1", "B2", "B3", "B4"]),
    #            (Meteor(), "M"), (Rouge(), "R")]

    scorers = [(Cider(), "C"),
                (Bleu(4), ["B1", "B2", "B3", "B4"]),
                (Rouge(), "R")]

    logger.info(f"loading ground-truths from {gt_caption}")
    with open(gt_caption) as f:
        gt_captions = json.load(f)
    # gt_captions = ptb_tokenizer.tokenize(gt_captions)

    logger.info(f"loading predictions from {pd_caption}")
    with open(pd_caption) as f:
        pd_captions = json.load(f)
    # pd_captions = ptb_tokenizer.tokenize(pd_captions)

    # pred = {"COCO_val2014_000000391895.jpg": ["a man riding a motorbike down a road with a bridge in the background."], 
    #         "COCO_val2014_000000060623.jpg": ["a woman sitting at a table with a plate of food sitting in front of her."], 
    #         "COCO_val2014_000000483108.jpg": ["a man riding a bike down a street next to a train with a cell phone."]}
        
    # gt = {"COCO_val2014_000000391895.jpg": ["a man with a red helmet on a small moped on a dirt road.", 
    #                                         "man riding a motor bike on a dirt road on the countryside.", 
    #                                         "a man riding on the back of a motorcycle.", 
    #                                         "a dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud wreathed mountains.", 
    #                                         "a man in a red shirt and a red hat is on a motorcycle on a hill side."], 
    #     "COCO_val2014_000000060623.jpg": ["a young girl inhales with the intent of blowing out a candle.", 
    #                                         "a young girl is preparing to blow out her candle.", 
    #                                         "a kid is to blow out the single candle in a bowl of birthday goodness.", 
    #                                         "girl blowing out the candle on an ice cream.", 
    #                                         "a little girl is getting ready to blow out a candle on a small dessert."], 
    #     "COCO_val2014_000000483108.jpg": ["a man on a bicycle riding next to a train.", 
    #                                         "a person is riding a bicycle but there is a train in the background.", 
    #                                         "a red and white train and a man riding a bicycle.", 
    #                                         "a guy that is riding his bike next to a train.", 
    #                                         "a man riding a bike past a train traveling along tracks."]}

    # pred = ptb_tokenizer.tokenize(pred)
    # gt = ptb_tokenizer.tokenize(gt)

    # score, score_list = Cider().compute_score(gt_captions, pd_captions)
    # print(score)
    # exit(0)

    logger.info("Start evaluating")
    # score_all_level = list()
    # for level, v in pd_captions.items():
    scores = {}
    for (scorer, method) in scorers:
        score, score_list = scorer.compute_score(gt_captions, pd_captions)
        if type(score) == list:
            for m, s in zip(method, score):
                scores[m] = s
        else:
            scores[method] = score
        # if method == "C":
        #     score_all_level.append(np.asarray(score_list))

    logger.info(
        ' '.join([
            "C: {C:.4f}",
            "R: {R:.4f}",
            "B1: {B1:.4f}", "B2: {B2:.4f}",
            "B3: {B3:.4f}", "B4: {B4:.4f}"
        ]).format(
            C=scores['C'],
            R=scores['R'],
            B1=scores['B1'], B2=scores['B2'],
            B3=scores['B3'], B4=scores['B4']
        ))

    # score_all_level = np.stack(score_all_level, axis=1)
    # logger.info(
    #     '  '.join([
    #         "4 level ensemble CIDEr: {C4:.4f}",
    #         "3 level ensemble CIDEr: {C3:.4f}",
    #         "2 level ensemble CIDEr: {C2:.4f}",
    #     ]).format(
    #         C4=score_all_level.max(axis=1).mean(),
    #         C3=score_all_level[:, :3].max(axis=1).mean(),
    #         C2=score_all_level[:, :2].max(axis=1).mean(),
    #     ))

# Cider: 0.5434 
# Rouge: 0.4815 
# B1: 0.5875 
# B2: 0.4346 
# B3: 0.3065 
# B4: 0.2136