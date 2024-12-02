#!/bin/bash

echo "Create results file"
python scripts/verify_answers.py
python scripts/create_result_file.py