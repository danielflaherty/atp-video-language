from google.cloud import translate_v2 as translate
import os
import csv
import random
from tqdm import tqdm
from google.cloud import translate_v2 as translate
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess all vids by extracting frames, and then storing CLIP embeddings of frames')
    parser.add_argument('--input_file', type=str, help='Input csv which should have all video urls')
    parser.add_argument('--output_file', type=str, help='', default="egoSchema_translated_new.csv")
    args = parser.parse_args()
    # Set the path to the service account key file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/danielflaherty/atp-video-language/data/egoSchema/egoschema-f8f502cc18dc.json'

    # Define the set of languages
    languages = ['ar', 'hi', 'ja', 'kk', 'ms', 'pl', 'pt', 'es', 'it', 'hu', 'fr', 'cs', 'fa', 'sr', 'no', 'sw', 'tr', 'vi', 'cy', 'th']

    # Read the csv file
    df = pd.read_csv(args.input_file)

    # Create a new DataFrame to store translated texts
    translated_rows = []

    # Instantiate a translation client
    translate_client = translate.Client()

    # Go through each row and translate each string
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        translated_row = {"video_id": row["video_id"], "question": row["question"]}
        for col in ["correct_answer", "wrong_answer_1", "wrong_answer_2", "wrong_answer_3", "wrong_answer_4"]:
            # Randomly select a language
            lang = random.choice(languages)

            # Translate to random language and back to English
            translation = translate_client.translate(row[col], target_language=lang)
            translated = translation['translatedText']

            back_to_english = translate_client.translate(translated, target_language='en')
            final_text = back_to_english['translatedText']

            translated_row[col] = final_text

        translated_rows.append(translated_row)

    # Write the translated texts to a new csv file
    headers = ["video_id", "question", "correct_answer", "wrong_answer_1", "wrong_answer_2", "wrong_answer_3", "wrong_answer_4"]
    with open(args.output_file, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)

        # Write the headers
        writer.writeheader()

        # Write the dictionaries as rows
        for row in translated_rows:
            writer.writerow(row)


