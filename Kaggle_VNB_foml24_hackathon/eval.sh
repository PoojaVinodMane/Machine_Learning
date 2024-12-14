#!/bin/bash
ROLLNO=$1
jupyter nbconvert --to script "cs22btech11035_foml24_hackathon.ipynb"
python3 cs22btech11035_foml24_hackathon.py --train-file train.csv --test-file FoML24/data/foml_hackathon_public_test.csv --predictions-file cs22btech11035_submission.csv
