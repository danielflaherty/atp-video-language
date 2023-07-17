import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def split_csv_file(csv_file, split_ratio, amount, output_dir, output_name):
    # Load the CSV file using pandas
    df = pd.read_csv(csv_file)
    df = df.sample(n=amount)
    
    # Split the data into training and validation sets using the specified split ratio
    train_data, val_data = train_test_split(df, test_size=split_ratio)
    
    # Write the training data to the "egoSchema_train.csv" file
    train_path = os.path.join(output_dir, "{}_train.csv".format(output_name))
    train_data.to_csv(train_path, index=False)
    
    # Write the validation data to the "egoSchema_val.csv" file
    val_path = os.path.join(output_dir, "{}_val.csv".format(output_name))
    val_data.to_csv(val_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess all vids by extracting frames, and then storing CLIP embeddings of frames')
    parser.add_argument('--input_file', type=str, help='Input csv which should have all video urls')
    parser.add_argument('--output_dir', type=str, default='/home/danielflaherty/atp-video-language/data/egoSchema/new_mixed_data', help='Output directory to dump mixed data train & val files in')
    parser.add_argument('--output_name', type=str, help='Output directory to dump mixed data train & val files in')
    parser.add_argument('--split_ratio', type=int, default=0.2)
    parser.add_argument('--amount', type=int, default=200)
    args = parser.parse_args()

    split_csv_file(args.input_file, args.split_ratio, args.amount, args.output_dir, args.output_name)