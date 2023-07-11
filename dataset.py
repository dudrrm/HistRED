import logging
import os
import os.path as osp
import json
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from utils import seed_worker
import pprint


def load_data(args,
              config=None, config_kor=None, config_han=None,
              tokenizer=None, tokenizer_kor=None, tokenizer_han=None,
              split="train"):

    if args.joint:
        dataset = JointDataset(args, config_kor, config_han, tokenizer_kor, tokenizer_han, split)
        
    else:
        assert args.language in ['korean', 'hanja']

        if args.language == 'korean':
            dataset = KoreanDataset(args, config, tokenizer, split)
        elif args.language == 'hanja':
            dataset = HanjaDataset(args, config, tokenizer, split)

    if split == "train":
        dataloader = DataLoader(dataset,
                                batch_size=args.train_batch_size,
                                collate_fn=dataset.collate_fn,
                                worker_init_fn=seed_worker,
                                num_workers=args.num_workers,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True)
    elif split == "valid":
        dataloader = DataLoader(dataset,
                                batch_size=args.eval_batch_size,
                                collate_fn=dataset.collate_fn,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
    elif split =="test":
        dataloader = DataLoader(dataset,
                                batch_size=args.test_batch_size,
                                collate_fn=dataset.collate_fn,
                                shuffle=False,
                                drop_last=False)
    else:
        raise ValueError("Data split must be either train/valid/test.")
    
    return dataloader


class JointDataset(Dataset):

    def __init__(self, args, config_kor, config_han, tokenizer_kor, tokenizer_han, split="train"):
        self.args = args
        self.config_kor = config_kor
        self.config_han = config_han
        self.tokenizer_kor = tokenizer_kor
        self.tokenizer_han = tokenizer_han
        self.split = split
        self.features = []
        
        if args.add_emb:
            self.save_dir = osp.join(args.data_dir, f"joint_add_{args.w_kor_emb}")
        else:
            self.save_dir = osp.join(args.data_dir, "joint_concat")

        self.save_path = osp.join(self.save_dir, f"{args.model_type}+{args.model2_type}_{split}.pt")
        os.makedirs(self.save_dir, exist_ok=True)

        map_dir = '/'.join(args.data_dir.split('/')[:-1])

        with open(osp.join(map_dir, "ner_map.json")) as f:
            self.ner_map = json.load(f)
        with open(osp.join(map_dir, "label_map.json")) as f:
            self.label_map = json.load(f)

        self.load_and_cache_examples()


    def load_and_cache_examples(self):
        if osp.exists(self.save_path):
            logging.info(f"Loading features from {self.save_path}")
            self.features = torch.load(self.save_path)
            return

        cls_token_kor = self.tokenizer_kor.cls_token
        sep_token_kor = self.tokenizer_kor.sep_token
        cls_token_han = self.tokenizer_han.cls_token
        sep_token_han = self.tokenizer_han.sep_token
        num_special_tokens = 2
        num_empty_entity_examples = 0
        num_empty_label_examples = 0
        num_filtered_labels = 0
        
        logging.info(f"Creating features from {self.args.data_dir}")
        rootdir = osp.join(self.args.data_dir, f"{self.split}")

        N_data_problems = 0

        for json_file in tqdm(os.listdir(rootdir), desc="Converting examples to features"):
            with open(osp.join(rootdir, json_file), encoding='utf-8') as f:
                ex = json.load(f)

                if len(ex["entity"]) == 0:
                    num_empty_entity_examples += 1
                    continue

                if len(ex["relation"]) == 0:
                    num_empty_label_examples += 1
                    continue

                ### Tokenize text & cluster entity mentions ###
                entities_kor = []       # list of lists clustering same entity mentions
                entities_han = []
                coref_dict_kor = {}     # { coref_type: entity_idx } -> will be used to cluster mentions
                coref_dict_han = {}
                ent2idx_kor = {}        # { info: entity_idx } -> map entity to idx
                ent2idx_han = {}
                ent_idx_kor = 0         # unique entity idx
                ent_idx_han = 0
                prev_idx_kor = 1    # skip cls_token idx
                prev_idx_han = 1    
                input_tokens_kor = [cls_token_kor]
                input_tokens_han = [cls_token_han]
                long_seq = False

                for ent in ex["entity"]:
                    if (ent["kor"]["type"] == "START" or ent["kor"]["text"] == "" or ent["kor"]["text"] == " " or
                        ent["han"]["type"] == "START" or ent["han"]["text"] == "" or ent["han"]["text"] == " "):
                        continue

                    if ent["han"]["coref_type"] != ent["kor"]["coref_type"]:
                        ent["han"]["coref_type"] = ent["kor"]["coref_type"]
                    # when tokenizing, make note of subword idxes
                    prev_text_kor = ex["text"]["kor"][prev_idx_kor:ent["kor"]["start"]]
                    prev_text_han = ex["text"]["han"][prev_idx_han:ent["han"]["start"]]
                    prev_tokens_kor = self.tokenizer_kor.tokenize(prev_text_kor)
                    prev_tokens_han = self.tokenizer_han.tokenize(prev_text_han)
                    input_tokens_kor += prev_tokens_kor
                    input_tokens_han += prev_tokens_han
                    start_kor = len(input_tokens_kor)
                    start_han = len(input_tokens_han)
                    ent_text_kor = ex["text"]["kor"][ent["kor"]["start"]:ent["kor"]["end"]]
                    ent_text_han = ex["text"]["han"][ent["han"]["start"]:ent["han"]["end"]]
                    ent_tokens_kor = self.tokenizer_kor.tokenize(ent_text_kor)
                    ent_tokens_han = self.tokenizer_han.tokenize(ent_text_han)
                    if self.args.mark_entities:
                        ent_tokens_kor = ["*"] + ent_tokens_kor + ["*"]
                        ent_tokens_han = ["*"] + ent_tokens_han + ["*"]
                    input_tokens_kor += ent_tokens_kor
                    input_tokens_han += ent_tokens_han
                    end_kor = len(input_tokens_kor)
                    end_han = len(input_tokens_han)
                    prev_idx_kor = ent["kor"]["end"]
                    prev_idx_han = ent["han"]["end"]

                    if (start_kor > self.args.max_seq_length-num_special_tokens or
                        end_kor > self.args.max_seq_length-num_special_tokens or
                        start_han > self.args.max_seq_length-num_special_tokens or
                        end_han > self.args.max_seq_length-num_special_tokens):
                        long_seq = True
                        break

                    ent_info_kor = (ent["kor"]["text"], ent["kor"]["start"], ent["kor"]["end"])
                    ent_info_han = (ent["han"]["text"], ent["han"]["start"], ent["han"]["end"])
                    full_ent_info_kor = (ent["kor"]["text"], ent["kor"]["start"], ent["kor"]["end"], start_kor, end_kor)
                    full_ent_info_han = (ent["han"]["text"], ent["han"]["start"], ent["han"]["end"], start_han, end_han)

                    if ent["kor"]["coref_type"]:
                        if ent["kor"]["coref_type"] in coref_dict_kor:
                            coref_idx = coref_dict_kor[ent["kor"]["coref_type"]]
                            ent2idx_kor[ent_info_kor] = coref_idx
                            entities_kor[coref_idx].append(full_ent_info_kor)
                        else:
                            coref_dict_kor[ent["kor"]["coref_type"]] = ent_idx_kor
                            ent2idx_kor[ent_info_kor] = ent_idx_kor
                            entities_kor.append([full_ent_info_kor])
                            ent_idx_kor += 1
                    else:
                        ent2idx_kor[ent_info_kor] = ent_idx_kor
                        entities_kor.append([full_ent_info_kor])
                        ent_idx_kor += 1

                    if ent["han"]["coref_type"]:
                        if ent["han"]["coref_type"] in coref_dict_han:
                            coref_idx = coref_dict_han[ent["han"]["coref_type"]]
                            ent2idx_han[ent_info_han] = coref_idx
                            entities_han[coref_idx].append(full_ent_info_han)
                        else:
                            coref_dict_han[ent["han"]["coref_type"]] = ent_idx_han
                            ent2idx_han[ent_info_han] = ent_idx_han
                            entities_han.append([full_ent_info_han])
                            ent_idx_han += 1
                    else:
                        ent2idx_han[ent_info_han] = ent_idx_han
                        entities_han.append([full_ent_info_han])
                        ent_idx_han += 1

                if not long_seq:
                    remaining_text_kor = ex["text"]["kor"][prev_idx_kor:]
                    remaining_text_han = ex["text"]["han"][prev_idx_han:]
                    input_tokens_kor += self.tokenizer_kor.tokenize(remaining_text_kor)
                    input_tokens_han += self.tokenizer_han.tokenize(remaining_text_han)
                input_tokens_kor = input_tokens_kor[:self.args.max_seq_length - 1]
                input_tokens_han = input_tokens_han[:self.args.max_seq_length - 1]
                input_tokens_kor += [sep_token_kor]
                input_tokens_han += [sep_token_han]
                input_ids_kor = self.tokenizer_kor.convert_tokens_to_ids(input_tokens_kor)
                input_ids_han = self.tokenizer_han.convert_tokens_to_ids(input_tokens_han)

                # Pad to max length
                input_ids_kor += [self.config_kor.pad_token_id] * (self.args.max_seq_length - len(input_ids_kor))
                input_ids_han += [self.config_han.pad_token_id] * (self.args.max_seq_length - len(input_ids_han))
                assert len(input_ids_kor) == len(input_ids_han) == self.args.max_seq_length

                ### entity masks & NERs
                ent_pos_kor, ent_pos_han = [], []
                for ent in entities_kor:
                    ent_pos_kor.append([])
                    for ment in ent:
                        token_start, token_end = ment[3], ment[4]
                        ent_pos_kor[-1].append((token_start, token_end))
                for ent in entities_han:
                    ent_pos_han.append([])
                    for ment in ent:
                        token_start, token_end = ment[3], ment[4]
                        ent_pos_han[-1].append((token_start, token_end))

                # debug
                for ent_k, ent_h in zip(ent_pos_kor, ent_pos_han):
                    assert len(ent_k) ==  len(ent_h)

                
                ### labels ###
                labels = torch.zeros((len(entities_kor), len(entities_kor), self.config_kor.num_labels), dtype=torch.float32)
                for relation in ex["relation"]:
                    s1, o1 = relation["kor"]['subject_entity'], relation["kor"]['object_entity']
                    s2, o2 = relation["han"]['subject_entity'], relation["han"]['object_entity']
                    h_idx = ent2idx_kor.get((s1["text"], s1["start"], s1["end"]), None)
                    t_idx = ent2idx_kor.get((o1["text"], o1["start"], o1["end"]), None)
                    h_idx2 = ent2idx_han.get((s2["text"], s2["start"], s2["end"]), None)
                    t_idx2 = ent2idx_han.get((o2["text"], o2["start"], o2["end"]), None)
                    if h_idx is None or t_idx is None:
                        num_filtered_labels += 1
                        continue
                    
                    r_idx = self.label_map[relation["kor"]["label"]]
                    labels[h_idx, t_idx, r_idx] = 1
                
                for h in range(len(entities_kor)):
                    for t in range(len(entities_kor)):
                        if torch.all(labels[h][t] == 0):
                            labels[h][t][0] = 1

                self.features.append({
                    "input_ids_kor": input_ids_kor,
                    "input_ids_han": input_ids_han,
                    "ent_pos_kor": ent_pos_kor,
                    "ent_pos_han": ent_pos_han,
                    "labels": labels, 
                    "entities_kor": entities_kor,
                    "entities_han": entities_han, 
                    "text_kor": ex["text"]["kor"],
                    "text_han": ex["text"]["han"] 
                })

        logging.info(f"# of empty entity examples filtered: {num_empty_entity_examples}")
        logging.info(f"# of empty label examples filtered: {num_empty_label_examples}")
        logging.info(f"# of beyond-truncated-text labels filtered: {num_filtered_labels}")
        logging.info(f"Saving features to {self.save_path}")
        torch.save(self.features, self.save_path)


    def collate_fn(self, samples):
        input_ids_kor = [x["input_ids_kor"] for x in samples]
        input_ids_han = [x["input_ids_han"] for x in samples]
        ent_pos_kor = [x["ent_pos_kor"] for x in samples]
        ent_pos_han = [x["ent_pos_han"] for x in samples]
        labels = [x["labels"].view(-1, self.config_kor.num_labels) for x in samples]

        input_ids_kor = torch.tensor(input_ids_kor, dtype=torch.long)
        input_ids_han = torch.tensor(input_ids_han, dtype=torch.long)
        labels = torch.cat(labels, dim=0)

        if not self.args.do_analysis:        
            return {"input_ids_kor": input_ids_kor,
                    "input_ids_han": input_ids_han,
                    "ent_pos_kor": ent_pos_kor,
                    "ent_pos_han": ent_pos_han,
                    "labels": labels}

        elif self.args.do_analysis:
            
            entities_kor = [x["entities_kor"] for x in samples]
            entities_han = [x["entities_han"] for x in samples]
            text_kor = [x["text_kor"] for x in samples]
            text_han = [x["text_han"] for x in samples]

            return {"input_ids_kor": input_ids_kor,
                    "input_ids_han": input_ids_han,
                    "ent_pos_kor": ent_pos_kor,
                    "ent_pos_han": ent_pos_han,
                    "labels": labels, 
                    "entities_kor": entities_kor,
                    "entities_han": entities_han, 
                    "text_kor": text_kor,
                    "text_han": text_han 
                    }


    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


class KoreanDataset(Dataset):

    def __init__(self, args, config, tokenizer, split="train"):
        self.args = args
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.features = []

        self.save_dir = osp.join(args.data_dir, args.language)
        self.save_path = osp.join(self.save_dir, f"{args.model_type}_{split}.pt")
        os.makedirs(self.save_dir, exist_ok=True)        
        
        map_dir = '/'.join(args.data_dir.split('/')[:-1])

        with open(osp.join(map_dir, "ner_map.json")) as f:
            self.ner_map = json.load(f)
        with open(osp.join(map_dir, "label_map.json")) as f:
            self.label_map = json.load(f)

        self.load_and_cache_examples()

    def load_and_cache_examples(self):
        if osp.exists(self.save_path):
            logging.info(f"Loading features from {self.save_path}")
            self.features = torch.load(self.save_path)
            return

        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        num_special_tokens = 2
        num_empty_entity_examples = 0
        num_empty_label_examples = 0
        num_filtered_labels = 0
        
        logging.info(f"Creating features from {self.args.data_dir}")
        rootdir = osp.join(self.args.data_dir, f"{self.split}")

        for json_file in tqdm(os.listdir(rootdir), desc="Converting examples to features"):
            with open(osp.join(rootdir, json_file), encoding='utf-8') as f:
                ex = json.load(f)
                
            if len(ex["entity"]) == 0:
                num_empty_entity_examples += 1
                continue

            if len(ex["relation"]) == 0:
                num_empty_label_examples += 1
                continue

            ### Tokenize text & cluster entity mentions ###
            entities = []       # list of lists clustering same entity mentions
            coref_dict = {}     # { coref_type: entity_idx } -> will be used to cluster mentions
            ent2idx = {}        # { info: entity_idx } -> map entity to idx
            ent_idx = 0         # unique entity idx
            prev_idx = 1        # skip cls_token idx
            input_tokens = [cls_token]
            long_seq = False

            for ent in ex["entity"]:
                ent = ent['kor']
                if ent["type"] == "START" or ent["text"] == "" or ent["text"] == " ":
                    continue
                # when tokenizing, make note of subword idxes
                prev_text = ex["text"]["kor"][prev_idx:ent["start"]]
                prev_tokens = self.tokenizer.tokenize(prev_text)
                input_tokens += prev_tokens
                start = len(input_tokens)
                ent_text = ex["text"]["kor"][ent["start"]:ent["end"]]
                ent_tokens = self.tokenizer.tokenize(ent_text)
                if self.args.mark_entities:
                    ent_tokens = ["*"] + ent_tokens + ["*"]
                input_tokens += ent_tokens
                end = len(input_tokens)
                prev_idx = ent["end"]

                # Skip entity mentions that appear beyond the truncated text
                if (start > self.args.max_seq_length-num_special_tokens or
                    end > self.args.max_seq_length-num_special_tokens):
                    long_seq = True
                    break

                # this tuple will be used to identify entity
                ent_info = (ent["text"], ent["start"], ent["end"], ent["type"])
                full_ent_info = (ent["text"], ent["start"], ent["end"], start, end, ent["type"])

                if ent["coref_type"]:
                    if ent["coref_type"] in coref_dict:
                        coref_idx = coref_dict[ent["coref_type"]]
                        ent2idx[ent_info] = coref_idx
                        entities[coref_idx].append(full_ent_info)
                    else:
                        coref_dict[ent["coref_type"]] = ent_idx
                        ent2idx[ent_info] = ent_idx
                        entities.append([full_ent_info])
                        ent_idx += 1
                else:
                    ent2idx[ent_info] = ent_idx
                    entities.append([full_ent_info])
                    ent_idx += 1

            if not long_seq:
                remaining_text = ex["text"]["kor"][prev_idx:]
                input_tokens += self.tokenizer.tokenize(remaining_text)
            input_tokens = input_tokens[:self.args.max_seq_length - 1]  # truncation
            input_tokens += [sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            # Pad to max length to enable sparse attention in bigbird
            input_ids += [self.config.pad_token_id] * (self.args.max_seq_length - len(input_ids))
            assert len(input_ids) == self.args.max_seq_length
            
            ### entity masks & NERs
            ent_pos, ent_ner = [], []
            for ent in entities:
                ent_pos.append([])
                # ent_ner.append([])
                for ment in ent:
                    token_start, token_end = ment[3], ment[4]
                    ent_pos[-1].append((token_start, token_end))
                    # ent_ner[-1].append(ment[-1])

            ### labels ###
            labels = torch.zeros((len(entities), len(entities), self.config.num_labels), dtype=torch.float32)
            for relation in ex["relation"]:
                relation = relation['kor']
                s, o = relation['subject_entity'], relation['object_entity']
                h_idx = ent2idx.get((s["text"], s["start"], s["end"], s["type"]), None)
                t_idx = ent2idx.get((o["text"], o["start"], o["end"], o["type"]), None)
                if h_idx is None or t_idx is None:
                    num_filtered_labels += 1
                    continue
                r_idx = self.label_map[relation["label"]]
                labels[h_idx, t_idx, r_idx] = 1

            for h in range(len(entities)):
                for t in range(len(entities)):
                    if torch.all(labels[h][t] == 0):
                        labels[h][t][0] = 1
                        
            self.features.append({
                "input_ids": input_ids,
                "ent_pos": ent_pos,
                "labels": labels,
            })

        logging.info(f"# of empty entity examples filtered: {num_empty_entity_examples}")
        logging.info(f"# of empty label examples filtered: {num_empty_label_examples}")
        logging.info(f"# of beyond-truncated-text labels filtered: {num_filtered_labels}")
        logging.info(f"Saving features to {self.save_path}")
        torch.save(self.features, self.save_path)


    def collate_fn(self, samples):
        input_ids = [x["input_ids"] for x in samples]
        ent_pos = [x["ent_pos"] for x in samples]
        labels = [x["labels"].view(-1, self.config.num_labels) for x in samples]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.cat(labels, dim=0)
        
        return {"input_ids": input_ids,
                "ent_pos": ent_pos,
                "labels": labels,
                }

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]



class HanjaDataset(Dataset):

    def __init__(self, args, config, tokenizer, split="train"):
        self.args = args
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.features = []
        
        self.save_dir = osp.join(args.data_dir, args.language)
        self.save_path = osp.join(self.save_dir, f"{args.model_type}_{split}.pt")
        os.makedirs(self.save_dir, exist_ok=True)

        
        map_dir = '/'.join(args.data_dir.split('/')[:-1])

        with open(osp.join(map_dir, "ner_map.json")) as f:
            self.ner_map = json.load(f)
        with open(osp.join(map_dir, "label_map.json")) as f:
            self.label_map = json.load(f)

        self.load_and_cache_examples()


    def load_and_cache_examples(self):
        if osp.exists(self.save_path):
            logging.info(f"Loading features from {self.save_path}")
            self.features = torch.load(self.save_path)
            return

        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        num_special_tokens = 2
        num_empty_entity_examples = 0
        num_empty_label_examples = 0
        num_filtered_labels = 0
        
        logging.info(f"Creating features from {self.args.data_dir}")
        rootdir = osp.join(self.args.data_dir, f"{self.split}")

        for json_file in tqdm(os.listdir(rootdir), desc="Converting examples to features"):
            with open(osp.join(rootdir, json_file), encoding='utf-8') as f:
                ex = json.load(f)
                
            if len(ex["entity"]) == 0:
                num_empty_entity_examples += 1
                continue

            if len(ex["relation"]) == 0:
                num_empty_label_examples += 1
                continue
            ### Tokenize text & cluster entity mentions ###
            entities = []       # list of lists clustering same entity mentions
            coref_dict = {}     # { coref_type: entity_idx } -> will be used to cluster mentions
            ent2idx = {}        # { info: entity_idx } -> map entity to idx
            ent_idx = 0         # unique entity idx
            prev_idx = 1        # skip cls_token idx
            input_tokens = [cls_token]
            long_seq = False

            for ent in ex["entity"]:
                ent = ent['han']
                if ent["type"] == "START" or ent["text"] == "" or ent["text"] == " ":
                    continue
                # when tokenizing, make note of subword idxes
                prev_text = ex["text"]['han'][prev_idx:ent["start"]]
                prev_tokens = self.tokenizer.tokenize(prev_text)
                input_tokens += prev_tokens
                start = len(input_tokens)
                ent_text = ex["text"]['han'][ent["start"]:ent["end"]]
                ent_tokens = self.tokenizer.tokenize(ent_text)
                if self.args.mark_entities:
                    ent_tokens = ["*"] + ent_tokens + ["*"]
                input_tokens += ent_tokens
                end = len(input_tokens)
                prev_idx = ent["end"]

                # Skip entity mentions that appear beyond the truncated text
                if (start > self.args.max_seq_length-num_special_tokens or
                    end > self.args.max_seq_length-num_special_tokens):
                    long_seq = True
                    break

                # this tuple will be used to identify entity
                ent_info = (ent["text"], ent["start"], ent["end"], ent["type"])
                full_ent_info = (ent["text"], ent["start"], ent["end"], start, end, ent["type"])

                if ent["coref_type"]:
                    if ent["coref_type"] in coref_dict:
                        coref_idx = coref_dict[ent["coref_type"]]
                        ent2idx[ent_info] = coref_idx
                        entities[coref_idx].append(full_ent_info)
                    else:
                        coref_dict[ent["coref_type"]] = ent_idx
                        ent2idx[ent_info] = ent_idx
                        entities.append([full_ent_info])
                        ent_idx += 1
                else:
                    ent2idx[ent_info] = ent_idx
                    entities.append([full_ent_info])
                    ent_idx += 1

            if not long_seq:
                remaining_text = ex["text"]['han'][prev_idx:]
                input_tokens += self.tokenizer.tokenize(remaining_text)
            input_tokens = input_tokens[:self.args.max_seq_length - 1]  # truncation
            input_tokens += [sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            # Pad to max length to enable sparse attention in bigbird
            input_ids += [self.config.pad_token_id] * (self.args.max_seq_length - len(input_ids))
            assert len(input_ids) == self.args.max_seq_length
            
            ### entity masks & NERs
            ent_pos, ent_ner = [], []
            for ent in entities:
                ent_pos.append([])
                for ment in ent:
                    token_start, token_end = ment[3], ment[4]
                    ent_pos[-1].append((token_start, token_end))

            ### labels ###
            labels = torch.zeros((len(entities), len(entities), self.config.num_labels), dtype=torch.float32)
            for relation in ex["relation"]:
                r_idx = self.label_map[relation["label"]]
                relation = relation['han']
                s, o = relation['subject_entity'], relation['object_entity']
                h_idx = ent2idx.get((s["text"], s["start"], s["end"], s["type"]), None)
                t_idx = ent2idx.get((o["text"], o["start"], o["end"], o["type"]), None)
                if h_idx is None or t_idx is None:
                    num_filtered_labels += 1
                    continue
                labels[h_idx, t_idx, r_idx] = 1

            for h in range(len(entities)):
                for t in range(len(entities)):
                    if torch.all(labels[h][t] == 0):
                        labels[h][t][0] = 1

            self.features.append({
                "input_ids": input_ids,
                "ent_pos": ent_pos,
                "labels": labels,
            })

        logging.info(f"# of empty entity examples filtered: {num_empty_entity_examples}")
        logging.info(f"# of empty label examples filtered: {num_empty_label_examples}")
        logging.info(f"# of beyond-truncated-text labels filtered: {num_filtered_labels}")
        logging.info(f"Saving features to {self.save_path}")
        torch.save(self.features, self.save_path)

    def collate_fn(self, samples):
        input_ids = [x["input_ids"] for x in samples]

        ent_pos = [x["ent_pos"] for x in samples]
        labels = [x["labels"].view(-1, self.config.num_labels) for x in samples]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.cat(labels, dim=0)
        
        return {"input_ids": input_ids,
                "ent_pos": ent_pos,
                "labels": labels,
                }

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]