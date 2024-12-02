#!/bin/bash
filename='book'

echo "Create json dataset from md file"
python scripts/create_dataset.py --filename data/book/${filename}

echo "Reduce samples length to avoid OOM"
python scripts/preprocessing_data.py --filename data/book/${filename}

echo "Create train and test files"
python scripts/prepare_custom_dataset.py --data_file_name ${filename}_dataset.json

