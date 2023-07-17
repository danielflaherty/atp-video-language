import csv
import argparse
import re

def transform_csv(input_file_path, output_file_path):
    # Open the input CSV file
    with open(input_file_path, 'r') as input_file:
        reader = csv.reader(input_file)
        # Skip the header row
        next(reader)
        # Open the output CSV file
        with open(output_file_path, 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            # Write the header row for the output CSV file
            writer.writerow(['batch_name', 'video_id', 'question', 'correct_answer', 'wrong_answer_1', 'wrong_answer_2', 'wrong_answer_3', 'wrong_answer_4'])
            # Loop through each row in the input CSV file
            for i, row in enumerate(reader):
                batch_name = row[0]
                video_id = row[1]
                question_1 = row[3]
                # question_2 = row[4]
                # question_3 = row[5]
                # Check if each question is non-empty
                if question_1:
                    q1 = question_1.split("\n")
                    q1 = [a.split(": ")[1] for a in q1]
                    writer.writerow([batch_name, video_id] + q1)
                # if question_2:
                #     q2 = question_2.split("\n")
                #     q2 = [a.split(": ")[1] for a in q2]
                #     writer.writerow([video_id] + q2)
                # if question_3:
                #     q3 = question_3.split("\n")
                #     q3 = [a.split(": ")[1] for a in q3]
                #     writer.writerow([video_id] + q3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_file', type=str, help='')
    parser.add_argument('--output_file', type=str, default='prompt3_bard.csv')
    args = parser.parse_args()

    # Process input file and save output files
    transform_csv(args.input_file, args.output_file)