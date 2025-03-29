import pandas as pd
import json

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")

# Convert to JSONL format
with open("ncert_finetune.jsonl", "w") as f:
    for _, row in df.iterrows():
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful AI that answers NCERT questions."},
                {"role": "user", "content": f"Question: {row['Question']}"},
                {"role": "assistant", "content": f"Answer: {row['Answer']}\nExplanation: {row['Explanation']}"}
            ]
        }
        f.write(json.dumps(data) + "\n")

print("✅ Dataset converted and saved as ncert_finetune.jsonl")
