import torch as th
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import pandas as pd
import pickle
import math
import os
import numpy as np
import random


class MC_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        subtitles_path,
        features_path,
        max_feats=70,
        features_dim=512,
        tokenizer=None,
        with_atp=False,
        use_context=True,
        type_map=None,
        prefix="",
        suffix="",
        is_test=False,
    ):
        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.mask = tokenizer.mask_token if tokenizer is not None else None
        self.with_atp = with_atp
        self.mc = 5
        self.type_map = type_map
        self.prefix = prefix
        self.suffix = suffix
        self.is_test = is_test

        self.directories = ["/home/raiymbek/benchmarks/FrozenBiLM/features/batch1_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch4.2_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch2_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch3_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch4_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch5_sample_1_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch5_sample_2_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch5_sample_3_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch5_sample_4_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch5_set_1_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch5_set_2_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch5_set_3_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch5_set_4_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch6_0_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch6_1_90",
                            "/home/raiymbek/benchmarks/FrozenBiLM/features/batch7_0_90",]

        self.all_ans_choices = []
        if self.is_test:
            for idx, row in self.data.iterrows():
                self.all_ans_choices.append(row["correct_answer"].capitalize().strip())
                self.all_ans_choices.append(row["wrong_answer_1"].capitalize().strip())
                self.all_ans_choices.append(row["wrong_answer_2"].capitalize().strip())
                self.all_ans_choices.append(row["wrong_answer_3"].capitalize().strip())
                self.all_ans_choices.append(row["wrong_answer_4"].capitalize().strip())
            self.mc = len(self.all_ans_choices)


    def __len__(self):
        return len(self.data)

    """
    *** NOT USED ***
    """
    def _get_subtitles(self, video_id, start, end):
        # only consider subtitles that intersec with the timestamps of the video clip
        subs_list = [
            x["text"]
            for x in self.subs[video_id]
            if x["end"] >= start and x["start"] <= end
        ]
        return " ".join(subs_list).capitalize().strip()

    def _get_text(self, answer, mask, question=None):
        text = (
            f"{self.prefix} Question: {question} Is it '{answer}'? {mask}{self.suffix}"
        )
        text = text.strip()
        return text

    def _get_video(self, video_id):
        video = None
        file_name = str(video_id).strip() + ".npy"
        for directory in self.directories:
            for root, dirs, files in os.walk(directory):
                if file_name in files:
                    video = os.path.join(root, file_name)
                    break
            if video is not None:
                break
        video = th.from_numpy(np.load(video).astype("float32"))
        if self.with_atp:
            video_len = 1000
        else:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
            video_len = self.max_feats

        return video, video_len

    def __getitem__(self, idx):
        video_id = self.data["video_id"].values[idx]

        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
        type = 0
        if "type" in self.data:
            type = self.data["type"].values[idx]

        # get features
        video, video_len = self._get_video(video_id)

        correct_answer = self.data['correct_answer'].values[idx].capitalize().strip()
        answers = []
        if self.is_test:
            answers = self.all_ans_choices
        else:
            answers = [correct_answer]
            for i in range(4):
                answers.append(self.data['wrong_answer_{}'.format(i + 1)].values[idx].capitalize().strip())

        random.shuffle(answers)
        correct_answer_id = answers.index(correct_answer)
        text = []
        for answer in answers:
            text.append(self._get_text(answer, self.mask, question=question))

        qid = idx

        return {
            "video": video,
            "video_len": video_len,
            "text": text,
            "qid": qid,
            "answer_id": correct_answer_id,
            "type": type,
        }


def mc_collate_fn(batch):
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
    text = [
        [batch[i]["text"][j] for i in range(bs)] for j in range(len(batch[0]["text"]))
    ]
    qid = [batch[i]["qid"] for i in range(bs)]
    answer_id = default_collate([batch[i]["answer_id"] for i in range(bs)])
    type = [batch[i]["type"] for i in range(bs)]

    return {
        "video": video,
        "video_len": video_len,
        "text": text,
        "qid": qid,
        "answer_id": answer_id,
        "type": type,
    }


def build_mc_dataset(dataset_name, split, args, tokenizer):
    type_map = None
    is_test = False
    if dataset_name == "how2qa":
        if split == "train":
            csv_path = args.how2qa_train_csv_path
        elif split == "val":
            csv_path = args.how2qa_val_csv_path
        elif split == "test":
            csv_path = args.how2qa_val_csv_path  # eval on val public
        else:
            raise NotImplementedError
        subtitles_path = args.how2qa_subtitles_path
        features_path = args.how2qa_features_path
    elif dataset_name == "tvqa":
        if split == "train":
            csv_path = args.tvqa_train_csv_path
        elif split == "val":
            csv_path = args.tvqa_val_csv_path
        elif split == "test":
            csv_path = args.tvqa_test_csv_path
        else:
            raise NotImplementedError
        subtitles_path = args.tvqa_subtitles_path
        features_path = args.tvqa_features_path
    elif dataset_name == 'egoSchema':
        if split == "train":
            csv_path = args.egoSchema_train_csv_path
        elif split == "val":
            csv_path = args.egoSchema_val_csv_path
        elif split == "test":
            csv_path = args.egoSchema_test_csv_path 
            # is_test = args.create_new_egoSchema_qa 
        else:
            raise NotImplementedError
        subtitles_path = args.egoSchema_subtitles_path
        features_path = args.egoSchema_features_path
    else:
        raise NotImplementedError
    return MC_Dataset(
        csv_path=csv_path,
        subtitles_path=subtitles_path,
        features_path=features_path,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
        tokenizer=tokenizer,
        with_atp=args.with_atp,
        use_context=args.use_context,
        prefix=args.prefix,
        suffix=args.suffix,
        type_map=type_map,
        is_test=is_test,
    )
