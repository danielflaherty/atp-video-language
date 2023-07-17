import os
import json
import pickle
import re
import io
import cv2
import argparse
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import gdown
import multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method

try:
     set_start_method('spawn', force=True)
except RuntimeError:
    pass

device = "cuda:0"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def process_video(input_vals):
    index, video_url, output_folder, batch_size, n_frames = input_vals
    file_id = re.search('/file/d/(.*?)/', video_url).group(1)
    download_loc = "./download_placeholder/vid_placeholder{}.mp4".format(index)
    gdown.download(video_url, download_loc, quiet=False, fuzzy=True)
    cap = cv2.VideoCapture(download_loc)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted(random.sample(range(frame_count), n_frames))
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()

    # Remove file after extracting frames to save space
    os.remove(download_loc)

    # Split frames up into batch sized lists
    batched_frames = []
    for i in range(0, len(frames), batch_size):
        batched_frames.append(frames[i : i + batch_size])

    # Get CLIP Embedding of Frames
    img_embs = []
    for batch_frames in batched_frames:
        inputs = processor(text=["dummy"], images=batch_frames, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        img_embs.append(outputs['image_embeds'].detach().cpu().numpy())

    # Save Image Embeddings as .npy file
    img_embs = np.concatenate(img_embs, axis=0)
    np.save(output_folder + '{}.npy'.format(file_id), img_embs)
    

def process_csv(csv_file, output_folder, n_frames, batch_size):
    df = pd.read_csv(csv_file)
    # num_processes = mp.cpu_count()
    # print(num_processes)
    pool = Pool(6)
    with tqdm(total=len(df)) as pbar:
        for i, processed_video in enumerate(pool.imap(process_video, [(index, row["video_url"], output_folder, batch_size, n_frames) for index, row in df.iterrows()])):
            pbar.update()
    pool.close()
    pool.join()
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess all vids by extracting frames, and then storing CLIP embeddings of frames')
    parser.add_argument('--input_file', type=str, help='Input csv which should have all video urls')
    parser.add_argument('--output_folder', type=str, default='/home/danielflaherty/atp-video-language/data/vid_frame_embs/')
    parser.add_argument('--n_frames', type=int, default=32, help='Number of frames to extract from video')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for CLIP')
    args = parser.parse_args()

    # Process input file and save output files
    mp.freeze_support()
    process_csv(args.input_file, args.output_folder, args.n_frames, args.batch_size)