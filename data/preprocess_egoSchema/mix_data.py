import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mix data from different csv files')
    parser.add_argument('--input_files', nargs="+", default=[], help='Input csv paths')
    parser.add_argument('--input_amounts', nargs="+", default=[], help='Number of rows of each csv to use in mixed data')
    parser.add_argument('--split_ratio', type=int, default=0.2, help='Split ratio to use in train/val split')
    parser.add_argument('--output_dir', type=str, default='/home/danielflaherty/atp-video-language/data/egoSchema/new_mixed_data', help='Output directory to dump mixed data train & val files in')
    parser.add_argument('--output_name', type=str, help='Output directory to dump mixed data train & val files in')
    args = parser.parse_args()

    # create one DataFrame of mixed data with desired amounts
    sub_dfs = []
    for i in range(len(args.input_files)):
        df = pd.read_csv(args.input_files[i])
        amount = args.input_amounts[i]

        sample = df.sample(n=int(amount))
        sub_dfs.append(sample)
    mixed_data = pd.concat(sub_dfs)
    mixed_data = mixed_data.sample(frac=1)

    # Split the data into training and validation sets using the specified split ratio
    train_data, val_data = train_test_split(mixed_data, test_size=args.split_ratio)
    
    # Write the training data to the "egoSchema_train.csv" file
    train_path = os.path.join(args.output_dir, "{}_train.csv".format(args.output_name))
    train_data.to_csv(train_path, index=False)
    
    # Write the validation data to the "egoSchema_val.csv" file
    val_path = os.path.join(args.output_dir, "{}_val.csv".format(args.output_name))
    val_data.to_csv(val_path, index=False)
