import pandas as pd
import json

# Load the CSV
data = pd.read_csv('HealthTestData/updated_labeled_news.csv')

# Function to determine split based on tweetID
def determine_split(tweet_id):
    last_digit = tweet_id % 10
    if last_digit in [1, 2, 3, 4, 5, 6]:
        return "train"
    elif last_digit in [7, 8]:
        return "test"
    else:
        return "val"

# Process each record into the desired JSON structure
processed_data = []
for index, row in data.iterrows():
    formatted_time = pd.to_datetime(row['created_at']).strftime("%Y-%m-%d %H:%M:%S")
    split = determine_split(row['tweetID'])
    td_acc = 1 if row['FakeNews_LLM'] == row['fake_label'] else 0
    record = {
        "content": row['context'],
        "label": 1 if row['fake_label'] == 0 else 0,  # Invert the label
        "time": formatted_time,
        "source_id": row['tweetID'],
        "td_rationale": row['Why'],
        "td_pred": 1 if row['FakeNews_LLM'] == 0 else 0,  # Invert the prediction
        "td_acc": td_acc,
        "split": split
    }
    processed_data.append(record)

# Split data into train, val, test
train_data = [d for d in processed_data if d['split'] == 'train']
val_data = [d for d in processed_data if d['split'] == 'val']
test_data = [d for d in processed_data if d['split'] == 'test']

# Save data to JSON files
with open('HealthTestData/train.json', 'w') as f:
    json.dump(train_data, f, indent=4)

with open('HealthTestData/val.json', 'w') as f:
    json.dump(val_data, f, indent=4)

with open('HealthTestData/test.json', 'w') as f:
    json.dump(test_data, f, indent=4)
