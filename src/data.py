"""
Simple video-language dataset class file, for illustration purposes. See comments below for relevant highlights.
"""

import os
import random
import pandas as pd
import torch
from torch.utils import data

class VideoLanguageDataset(data.Dataset):
    """
    Example (simple) video-language dataset class, for illustration purposes for ATP training.
    """
    def __init__(self, args, **kwargs):
        super().__init__()
        self.data_path = args.data_path
        # self.split = split
        self.tokenizer = tokenizer
        self.mask = tokenizer.mask_token
        self.get_text_query = args.use_text_query
        self.get_text_cands = args.use_text_cands
        self.n_frames = args.n_frames

        self.metadata = pd.read_csv(os.path.join(self.data_path, "ego_schema_data.csv"))
        
    def __len__(self):
        return len(self.metadata)

    def _get_text(self, answer, mask, question=None):
        text = (
            f"{self.prefix} Question: {question} Is it '{answer}'? {mask}{self.suffix}"
        )
        text = text.strip()
        return text
        
    def __getitem__(self, index):
        """
        Assuming torch files for each of the features for simplicity; adjust to fit your extracted features.
        (e.g. numpy, hdf5, json, etc.)
        """
        info_i = self.metadata.iloc[index]
        
        # get random frame sample, without replacement
        video_id = info_i["video_id"]
        video_features = torch.from_numpy(np.load(self.data_path + str(video_id) + ".npy").astype("float32")).unsqueeze(0)  # (L_video, D_in); L_video >> L
        # frame_idxs_gt = torch.randperm(len(video_features))[:self.n_frames]
        # video_features_sampled = video_features[frame_idxs_gt]  # (L, D_in)
        
        # get other features / labels
        question = info_i['question'].capitalize()
        correct_answer = info_i['correct_answer'].capitalize().strip()
        answers = [correct_answer]
        for i in range(4):
            answers.append(info_i['wrong_answer_{}'.format(i + 1)].capitalize().strip())
        random.shuffle(answers)
        correct_answer_id = answers.index(correct_answer)
        text = []
        for answer in answers:
            text.append(self._get_text(answer, self.mask, question=question))
        # text_query_features = self.tokenizer([question], )
        # text_cands_features = torch.load(info_i["text_cands_features"]) if self.get_text_cands else []
        # labels_gt = torch.load(info_i["labels_gt"])
        
        return video_features, text, correct_answer_id
        
        
        
        
        

