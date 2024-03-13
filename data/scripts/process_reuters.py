import argparse
from collections import Counter
import os
import csv
from concurrent.futures import ProcessPoolExecutor

MAX_WORKERS = 4
MIN_FEATURE_PERCENTAGE = 0.1

def process_language(lang1, lang2, input_directory, output_directory):
    unique_features = Counter()
    feature_value_pairs = []
    categories = []

    output_path_csv = os.path.join(output_directory, f"ds_{lang1}", f"view_{lang2}.csv")

    with open(output_path_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Collect all unique features
        file_path = os.path.join(input_directory, lang2, f'Index_{lang1}-{lang2}')
        with open(file_path, 'r') as file:
            for line in file:
                if len(line.strip().split(' ', 1)) == 1:
                    continue
                features = line.strip().split(' ', 1)[1]
                line_feature_value_pairs = features.split(' ')
                feature_value_pairs.append(line_feature_value_pairs)
                categories.append(line.strip().split(' ', 1)[0])

                features = [int(pair.split(':')[0]) for pair in line_feature_value_pairs]
                unique_features.update(features)

            total_documents = len(categories)
            min_occurrences = MIN_FEATURE_PERCENTAGE * total_documents
            sorted_features = [feature for feature, count in unique_features.items() if count >= min_occurrences]
            print(f"{len(sorted_features)} features with at least {MIN_FEATURE_PERCENTAGE} occurrences")

            del(unique_features)

            # Write header to CSV
            header = ['Category'] + [str(feature) for feature in sorted_features]
            csv_writer.writerow(header)

            for cat, line in zip(categories, feature_value_pairs):
                row = [None] * (len(sorted_features) + 1)
                row[0] = cat
                for pair in line:
                    feature_index = int(pair.split(':')[0])
                    if feature_index in sorted_features:
                        feature_value = pair.split(':')[1]
                        row[sorted_features.index(feature_index) + 1] = feature_value
                csv_writer.writerow(row)

def transform_to_csv(input_directory, output_directory="/Reuters-rcv1-rcv2/"):
    languages = ['EN', 'FR', 'GR', 'IT', 'SP']
    os.makedirs(os.getcwd() + output_directory, exist_ok=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for lang1 in languages:
            output_lang_dir = os.getcwd() + output_directory + "ds_" + lang1
            os.makedirs(output_lang_dir, exist_ok=True)
            for lang2 in languages:
                futures.append(executor.submit(process_language, lang1, lang2, input_directory, os.getcwd() + output_directory))

        # Wait for all tasks to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir", default="~/Multi-View-Majority-Vote-Learning-Algorithms-Direct-Minimization-of-PAC-Bayesian-Bounds/rcv1rcv2aminigoutte", type=str, help="The input directory of Reuters RCV1/RCV2 dataset."
    )

    argv = parser.parse_args()

    transform_to_csv(argv.input_dir)
