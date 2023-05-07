import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def split_csv_file(csv_file, split_ratio):
    # Load the CSV file using pandas
    df = pd.read_csv(csv_file)
    
    # Split the data into training and validation sets using the specified split ratio
    train_data, val_data = train_test_split(df, test_size=split_ratio)
    
    # Write the training data to the "egoSchema_train.csv" file
    train_data.to_csv("egoSchema_train.csv", index=False)
    
    # Write the validation data to the "egoSchema_val.csv" file
    val_data.to_csv("egoSchema_val.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess all vids by extracting frames, and then storing CLIP embeddings of frames')
    parser.add_argument('--input_file', type=str, help='Input csv which should have all video urls')
    parser.add_argument('--split_ratio', type=int, default=0.2)
    args = parser.parse_args()

    split_csv_file(args.input_file, args.split_ratio)