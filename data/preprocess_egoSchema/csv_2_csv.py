import pandas as pd
import csv

if __name__ == '__main__':
    df_ref = pd.read_csv("/home/danielflaherty/atp-video-language/data/egoSchema/egoSchema_all_data.csv")
    df_new = pd.read_csv("/home/danielflaherty/atp-video-language/data/egoSchema/new_qa.csv")
    headers = ["video_id", "video_url", "question", "correct_answer", "wrong_answer_1", "wrong_answer_2", "wrong_answer_3", "wrong_answer_4"]
    with open("egoSchema_ad_matching_bilm_70_frames.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        new_row = []
        for i, row in df_new.iterrows():
            video_row = df_ref.loc[df_ref["correct_answer"] == row['correct_answer']]
            video_id = video_row['video_id'].iloc[0]
            # question = video_row['question'].iloc[0]
            writer.writerow({
                'video_id': video_id,
                'video_url': "",
                'question': row[0],
                'correct_answer': row[1],
                'wrong_answer_1': row[2],
                'wrong_answer_2': row[3],
                'wrong_answer_3': row[4],
                'wrong_answer_4': row[5],
            })

