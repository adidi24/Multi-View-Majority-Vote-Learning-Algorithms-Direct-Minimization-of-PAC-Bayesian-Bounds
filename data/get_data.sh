#!/usr/bin/env zsh

DATA_DIR="~/Multi-View-Majority-Vote-Learning-Algorithms-Direct-Minimization-of-PAC-Bayesian-Bounds/data"

VALID_DATASETS=("multiple-features" "reuters-en" "sample_data")

download_and_structure() {
    dataset_name="$1"

    # Check if dataset name is valid
    if [[ ! "${VALID_DATASETS[@]}" =~ $dataset_name ]]; then
        echo "Invalid dataset name: '$dataset_name'. Please choose from: '${VALID_DATASETS[@]}'."
        return 1
    fi

    if [[ "$dataset_name" = "reuters-en" ]]; then

        echo "Downloading '$dataset_name' ---------------------------------------"
        # wget https://lig-membres.imag.fr/grimal/data/ReutersEN.tar.gz
        wget  https://archive.ics.uci.edu/static/public/259/reuters+rcv1+rcv2+multilingual+multiview+text+categorization+test+collection.zip

        echo "Extracting '$dataset_name' ---------------------------------------"
        unzip reuters+rcv1+rcv2+multilingual+multiview+text+categorization+test+collection.zip
        tar -xjf rcv1rcv2aminigoutte.tar.bz2
        # tar -xvzf ReutersEN.tar.gz

        # echo "Structuring the data  ---------------------------------------"
        # python scripts/process_reuters.py

        echo "Cleaning the archived files ---------------------------------------"
        rm rcv1rcv2aminigoutte.tar.bz2 $DATA_DIR/reuters+rcv1+rcv2+multilingual+multiview+text+categorization+test+collection.zip

        echo "Successfully downloaded and structured '$dataset_name'."
    elif [[ "$dataset_name" = "multiple-features" ]]; then
        echo "Downloading '$dataset_name' ---------------------------------------"
        wget https://archive.ics.uci.edu/static/public/72/multiple+features.zip

        echo "Extracting '$dataset_name' ---------------------------------------"
        unzip multiple+features.zip
        tar -xvf mfeat.tar

        echo "Cleaning the archived files ---------------------------------------"
        rm mfeat-* mfeat.* multiple+features.zip

        echo "Structuring the data  ---------------------------------------"
        python scripts/process_mfeat.py

    elif [[ "$dataset_name" = "sample_data" ]]; then
        echo "Downloading '$dataset_name' ---------------------------------------"
        git clone https://github.com/goyalanil/PB-MVBoost.git && mv PB-MVBoost/sample_data ./sample_data && sudo rm -r PB-MVBoost

    else
        echo "Invalid dataset name: '$dataset_name'. Please use alphanumeric characters"
    fi
}

if [[ $# = 0 ]]; then
    echo "No dataset name provided. Please specify a dataset name or 'all' to download all."
elif [[ "$1" = "all" ]]; then
    for dataset in "${VALID_DATASETS[@]}"; do
        if [[ "$dataset" != "all" ]]; then
            download_and_structure "$dataset"
        fi
    done
else
    download_and_structure "$1"
fi
