import os
import pandas as pd

def process_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = [list(map(float, line.split())) for line in lines]

    labels = [i // 200 for i in range(2000)]

    df = pd.DataFrame(data)

    # Adding a column for labels
    df['label'] = labels

    return df

def save_to_csv(dataframe, output_file):
    dataframe.to_csv(output_file, index=False)

if __name__ == "__main__":
    feature_sets = ["mfeat-fou", "mfeat-fac", "mfeat-kar", "mfeat-pix", "mfeat-zer", "mfeat-mor"]

    for feature_set in feature_sets:
        input_file = os.getcwd() + f"/mfeat/{feature_set}"
        output_csv = os.getcwd() + f"/mfeat/{feature_set}.csv"

        df = process_dataset(input_file)
        save_to_csv(df, output_csv)

        print(f"Processed {feature_set}.txt and saved to {feature_set}.csv")