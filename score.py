# f1_score.py
import argparse
import os 
import sys
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import datasets
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--outputs_folder", type=str, default="")
parser.add_argument("--dataset", type=str, default="")


def character_f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)

def convert_text_to_offsets(text, context, labels=False):
    subsets = text.split("; ")
    idxs = []
    for subset in subsets:
        start_idx = context.find(subset)
        if start_idx != -1:
            end_idx = start_idx + len(subset)
            idxs.extend(list(range(start_idx, end_idx)))
        elif subset == "N/A":
            pass
        elif labels:
            raise ValueError(f"Label {subset} not found in context {context}")

    return idxs

def f1_toxicspans(outputs, labels, srcs):
    f1_scores = []
    for i in range(len(outputs)):
        src_text = srcs[i]
        pred_text = outputs[i]
        gold_text = labels[i]

        pred_offsets = convert_text_to_offsets(pred_text, src_text)
        gold_offsets = convert_text_to_offsets(gold_text, src_text, labels=True)
        f1_scores.append(character_f1(pred_offsets, gold_offsets))
        # print(f"F1 score: {f1_score}")

    print(f"{100 * np.mean(f1_scores):.2f}")



def score_jigsaw(outputs, labels):

    average_f1 = []
    label_names = sorted(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'none of the above'])

    for i in range(len(outputs)):
        pred_text = sorted(outputs[i].split(", "))
        gold_text = sorted(labels[i].split(", "))
        f1_scores = []
        for label in label_names:
            pred_offset = pred_text.index(label)
            gold_offset = gold_text.index(label)
            f1_scores.append(character_f1(pred_offset, gold_offset))
        average_f1.append(np.mean(f1_scores))
    print(f"Average F1 score: {np.mean(average_f1)}")


def binary_jigsaw(outputs, labels):
    preds = []
    golds = []
    for i in range(len(outputs)):
        pred_text = "toxic" in outputs[i]
        gold_text = "toxic" in labels[i]
        preds.append(pred_text)
        golds.append(gold_text)
    accuracy = accuracy_score(preds, golds)
    print(f"Binary Accuracy: {accuracy}")
    fscore = f1_score(preds, golds, average="macro")
    print(f"Binary F1 score: {fscore}")


def read_file_to_array(file_path):
    with open(file_path, "r") as f:
        return f.readlines()


def f1(labels, outputs):
    possible_labels = list(set(labels))
    score = f1_score(labels, outputs, average="macro", labels=possible_labels)
    return score


def main(args):
    # outputs = read_file_to_array(os.path.join(args.outputs_folder, f"outputs_{args.dataset}.txt"))
    # labels = read_file_to_array(os.path.join(args.outputs_folder, f"labels_{args.dataset}.txt"))
    if args.dataset == 'hatecheck':
        pass
    else:
        outputs = pd.read_csv(os.path.join(args.outputs_folder, f"{args.dataset}.csv"), keep_default_na=False)['output'].tolist()
        labels = pd.read_csv(os.path.join(args.outputs_folder, f"{args.dataset}.csv"), keep_default_na=False)['label'].tolist()
        if 'toxicspans' in args.dataset:
            srcs = pd.read_csv(os.path.join(args.outputs_folder, f"{args.dataset}.csv"), keep_default_na=False)['source'].tolist()


    # print(len(outputs), len(labels))
    # print(outputs[:10])
    # print(labels[:10])

    # calculate f1 score 
    if 'toxicspans' in args.dataset:
        f1_toxicspans(outputs, labels, srcs)
    elif 'jigsaw' in args.dataset:
        binary_jigsaw(outputs, labels)
    elif 'hatecheck' in args.dataset:
        for lang in ['ara', 'cmn', 'deu', 'eng', 'fra', 'hin', 'ita', 'nld', 'pol', 'por', 'spa']:
            outputs = pd.read_csv(os.path.join(args.outputs_folder, f"{args.dataset}_{lang}_1.csv"), keep_default_na=False)['output'].tolist()
            labels = pd.read_csv(os.path.join(args.outputs_folder, f"{args.dataset}_{lang}_1.csv"), keep_default_na=False)['label'].tolist()

            score = f1(labels, outputs)
            # print(f"F1 score for {lang}: {score}")
            print(f"{score}")

    else:
        score = f1(labels, outputs)
        print(f"{100 * score:.2f}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)