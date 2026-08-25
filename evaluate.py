from unsloth import FastLanguageModel
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from functools import partial
from tqdm import tqdm
from collections import defaultdict
import os
import csv
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*resized since it had shape.*"
)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="data/combined/")
# --subset == --subtask 
parser.add_argument("--subset", "--subtask", type=str, default="all")
parser.add_argument("--model_type", type=str, default="qwen")
parser.add_argument("--regex", action="store_true", help="Use regex to filter datasets")
parser.add_argument("--batch_size", type=str, default=8)
parser.add_argument("--model_path", type=str, default="~/nobackup/models/Qwen3-30B-A3B-Instruct-2507/")
parser.add_argument("--output_path", type=str, default="")


def padding_collate_fn(batch, max_len=2048, skip_fields=[], add_labels=False, pad_token_id=0):
    """
        Pads each list with zeros and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    padded_batch = {}

    if add_labels:
        for b in range(len(batch)):
            batch[b]['labels'] = batch[b]['input_ids']

    for key in batch[0]:
        if key in skip_fields:
            padded_batch[key] = []
            continue
        largest = min(max_len, max([len(b[key]) for b in batch]))
        padded_batch[key] = torch.full((len(batch), largest), pad_token_id, dtype=torch.long)
        if "labels" in key:
            padded_batch[key] = torch.full((len(batch), largest), -100, dtype=torch.long)
        elif "attention_mask" in key:
            padded_batch[key] = torch.full((len(batch), largest), 0, dtype=torch.long)

    for i, sample in enumerate(batch):
        for key in padded_batch:
            if key in skip_fields:
                padded_batch[key].append(batch[i][key])
                continue
            key_len = min(max_len, len(sample[key]))
            padded_batch[key][i, -key_len:] = torch.LongTensor(sample[key][:key_len])

    return padded_batch


def preprocess_data(examples, tokenizer):
    tokenized = {
        'input_ids1': [], 
        'input_ids2': [],
        'input_ids3': [],
        'input_ids4': [],
        'input_ids5': [],
        'input_ids6': [],
        'attention_mask1': [],
        'attention_mask2': [],
        'attention_mask3': [],
        'attention_mask4': [],
        'attention_mask5': [],
        'attention_mask6': [],
        'labels1': [],
        'labels2': [],
        'labels3': [],
        'labels4': [],
        'labels5': [],
        'labels6': [],
    }

    # label_token = tokenizer.encode("Label: ", add_special_tokens=False, return_tensors='pt')[0]
    for i in range(len(examples['messages'])):
        assert len(examples['messages'][i]) <= 12, f"Num messages = {len(examples['messages'][i])}"
        for j in range(6):
            if j < len(examples['messages'][i]) // 2:
                src = tokenizer.apply_chat_template(examples['messages'][i][:j*2+1], tokenize=True, return_tensors='pt', add_generation_prompt=True)[0]
                # src = torch.cat((src, label_token))
                tgt = tokenizer.encode(examples['messages'][i][j*2+1]['content'], add_special_tokens=False, return_tensors='pt')[0]
            else:
                src = torch.tensor([-1])
                tgt = torch.tensor([-1])

            tokenized[f'input_ids{j+1}'].append(src)
            tokenized[f'labels{j+1}'].append(tgt)
            tokenized[f'attention_mask{j+1}'].append(torch.ones_like(src))

    return tokenized

def move_dict(d, device="cuda:0", skip_fields=[]):
    return {key: value.to(device=device) for key, value in d.items() if key not in skip_fields}

def evaluate_hierarchical(model, tokenizer, dataloader, output_path=None, args=None):
    model.eval()
    correct = defaultdict(int)
    total = defaultdict(int)

    correct_tokens = 0
    total_tokens = 0
    # avg_loss = 0

    outputs_str = {}
    labels_str = {}
    srcs_str = {}
    logits_str = {}

    eos_token = tokenizer.eos_token

    if args.model_type == "qwen":
        user_token = "<|im_start|>" if tokenizer.eos_token == "<|eot_id|>" else "<|start_header_id|>"
    elif args.model_type == "llama":
        user_token = "<|start_header_id|>"
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")


    # loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch.pop("seq_lengths", None)
            cuda_batch = move_dict(batch, skip_fields=['dataset', 'messages'])
            # print(batch)
            for cstep in range(6):
                labels = cuda_batch.pop(f"labels{cstep+1}", None)

                filtered_batch = {}
                filtered_batch['input_ids'] = cuda_batch.pop(f"input_ids{cstep+1}", None)
                filtered_batch['attention_mask'] = cuda_batch.pop(f"attention_mask{cstep+1}", None)

                # filter out rows where labels contains -1
                mask = labels[:, -1] != -1
                if mask.sum() == 0:
                    break

                # Get original indices that pass the filter
                original_indices = torch.where(mask)[0].cpu().tolist()

                filtered_batch["input_ids"] = filtered_batch["input_ids"][mask]
                filtered_batch["attention_mask"] = filtered_batch["attention_mask"][mask].clamp(min=0, max=1)
                labels = labels[mask]

                # print(filtered_batch["input_ids"].shape)
                # Synchronize CUDA operations to avoid CUDA graph errors with dynamic batch sizes
                torch.cuda.synchronize()
                outputs = model.generate(filtered_batch["input_ids"], 
                attention_mask=filtered_batch["attention_mask"],
                return_dict_in_generate=True, 
                output_logits=True,
                pad_token_id=tokenizer.pad_token_id,
                # temperature=0.7, # does not change anything
                # top_p=0.8,
                # top_k=20,
                # min_p=0,
                # force_words_ids=options,
                max_new_tokens=labels.shape[1]*2+3)

                logits = torch.stack(outputs.logits, dim=1)
                preds = logits.argmax(dim=-1)

                # same_len_logits = torch.zeros(logits.shape[0], max(logits.shape[1], labels.shape[1]), logits.shape[-1], device=logits.device)
                # same_len_logits[:, :logits.shape[1]] = logits
                # same_len_logits = same_len_logits[:, :labels.shape[1]]

                # avg_loss += loss_fn(same_len_logits.reshape(-1, same_len_logits.shape[-1]), labels.view(-1)).item()

                for b in range(len(filtered_batch["input_ids"])):
                    inp = filtered_batch["input_ids"][b]
                    inp = inp[inp != tokenizer.pad_token_id]
                    lab = labels[b][labels[b] != -100]
                    pre = preds[b][preds[b] != tokenizer.pad_token_id]
                    # remove 
                    lab = tokenizer.decode(lab, skip_special_tokens=True).strip()
                    # prehuh = tokenizer.decode(pre, skip_special_tokens=False)
                    # print(f"\033[96m{prehuh}\033[0m")
                    pre = tokenizer.decode(pre, skip_special_tokens=False).split(eos_token)[0].split(user_token)[0].strip()
                    inp = tokenizer.decode(inp, skip_special_tokens=True).strip()
                    # print in blue
                    # print(f"\033[94m{inp}\033[0m")

                    # Use original batch index instead of filtered index
                    orig_idx = original_indices[b]
                    name = f"{batch['dataset'][orig_idx]}_{cstep+1}"
                    if name not in outputs_str:
                        outputs_str[name] = []
                        labels_str[name] = []
                        srcs_str[name] = []
                        logits_str[name] = []
                    outputs_str[name].append(pre)
                    labels_str[name].append(lab)
                        
                    src_text = "\n\n".join(batch['messages'][orig_idx][0]['content'].split("\n\n")[1:-1]).strip()
                    src_text = src_text.split("\n\nWhich of the following categories")[0].strip()
                    src_text = src_text.split("\n\nWhich category does the text belong to?")[0].strip()
                    srcs_str[name].append(src_text)
                    # print(f"\033[94m{inp}\033[0m")
                    # print(f"\033[92m{lab}\033[0m")
                    # print(f"\033[91m{pre}\033[0m")
                    # print("__________________________________________")

                    correct[name] += (pre == lab)
                    total[name] += 1

                    correct_tokens += (pre == lab)
                    total_tokens += 1

    # save outputs and labels to csv files per dataset
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    for dataset in outputs_str:
        with open(os.path.join(output_path, f"{dataset}.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["output", "label", "source"])
            for i in range(len(outputs_str[dataset])):
                writer.writerow([outputs_str[dataset][i], labels_str[dataset][i], srcs_str[dataset][i]])

    results = {}
    for dataset in correct:
        results[dataset] = {'acc': 100 * correct[dataset] / total[dataset]}

    results['Overall'] = {'acc': 100 * correct_tokens / total_tokens} #'gen_loss': avg_loss / len(dataloader)}
    return results

def main():
    
    args = parser.parse_args()

    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # model = AutoModelForCausalLM.from_pretrained(args.model_path).to("cuda:0")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    dataset = load_from_disk(args.dataset)['test']
    # print(dataset.unique("dataset"))
    if args.regex:
        dataset = dataset.filter(lambda x: args.subset in x['dataset'], num_proc=10)
    elif args.subset != "all":
        subsets = args.subset.split(",")
        dataset = dataset.filter(lambda x: x['dataset'] in subsets, num_proc=10)
    else:
        dataset = dataset.filter(lambda x: x['dataset'] not in ["jigsaw"], num_proc=10)

    if 'text' in dataset.column_names:
        dataset = dataset.remove_columns(["text"])
    if 'label' in dataset.column_names:
        dataset = dataset.remove_columns(["label"])

    dataset = dataset.map(preprocess_data, 
        fn_kwargs={'tokenizer': tokenizer}, 
        batched=True)

    print(dataset)
    # print(dataset[0])

    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=args.batch_size, 
        collate_fn=partial(padding_collate_fn, skip_fields=['dataset', 'messages'], pad_token_id=tokenizer.pad_token_id), 
        shuffle=False)

    results = evaluate_hierarchical(model, tokenizer, dataloader, args.output_path, args)

    for dataset in results:
        print(f"{dataset}: {results[dataset]['acc']:.2f}")

    # print(f"Loss: {results['Overall']['gen_loss']:.4f}")


if __name__ == "__main__":
    main()