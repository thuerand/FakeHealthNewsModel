import pandas as pd
import json

# Load the CSV
data = pd.read_csv('data_corona/Analyzed_Fake_News_Dataset_Bereinigt.csv')

# Function to determine split based on the last digit of source_id
def determine_split(source_id):
    last_digit = int(str(source_id)[-1])
    if last_digit in [1, 2, 3, 4, 5, 6]:
        return "train"
    elif last_digit in [7, 8]:
        return "test"
    else:
        return "val"

# Process each record into the desired JSON structure
processed_data = []
for index, row in data.iterrows():
    split = determine_split(row['source_id'])
    record = {
        "content": row['content'],
        "label": 1 if row['label'] == 0 else 0,  # Adjust labels accordingly
        "time": "2018-12-17 12:30:36",  # Static time for all records
        "source_id": row['source_id'],
        "td_rationale": row['td_rationale'],
        "td_pred": 1 if row['td_pred'] == 0 else 0,
        "td_acc": row['td_acc'],
        "cs_rationale": row['cs_rationale'],
        "cs_pred": 1 if row['cs_pred'] == 0 else 0,
        "cs_acc": row['cs_acc'],
        "split": split
    }
    processed_data.append(record)

# Split data into train, val, test based on the 'split' tag
train_data = [d for d in processed_data if d['split'] == 'train']
val_data = [d for d in processed_data if d['split'] == 'val']
test_data = [d for d in processed_data if d['split'] == 'test']

# Save data to JSON files
with open('data_corona/train.json', 'w') as f:
    json.dump(train_data, f, indent=4)

with open('data_corona/val.json', 'w') as f:
    json.dump(val_data, f, indent=4)

with open('data_corona/test.json', 'w') as f:
    json.dump(test_data, f, indent=4)
