import pandas as pd
import json
import together

# Step 1: Convert CSV to JSONL for fine-tuning
def convert_csv_to_jsonl(csv_file, jsonl_file):
    df = pd.read_csv(csv_file)
    
    with open(jsonl_file, "w") as f:
        for _, row in df.iterrows():
            data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI that answers NCERT questions."},
                    {"role": "user", "content": f"Question: {row['Question']}"},
                    {"role": "assistant", "content": f"Answer: {row['Answer']}\nExplanation: {row['Explanation']}"}
                ]
            }
            f.write(json.dumps(data) + "\n")
    
    print(f"✅ Converted {csv_file} to {jsonl_file}")

# Step 2: Upload dataset to Together.ai
def upload_dataset(api_key, jsonl_file):
    client = together.Together(api_key=api_key)
    
    response = client.files.upload(file=jsonl_file)  # Corrected method

    print("✅ Upload response:", response)
    return response["id"]  # Ensure this key exists in the response

# Step 3: Fine-tune LLaMA-3 model
def fine_tune_model(api_key, file_id):
    client = together.Together(api_key=api_key)
    response = client.finetune(
        model="meta-llama/Llama-3-8b-chat-hf",
        training_file=file_id,
        n_epochs=3  # Change as needed
    )
    finetuned_model_id = response["id"]  # Ensure correct key
    print(f"✅ Fine-tuning started. Model ID: {finetuned_model_id}")
    return finetuned_model_id

# Step 4: Use the fine-tuned model
def chat_with_model(api_key, model_id, query):
    client = together.Together(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message["content"]

# Run the pipeline
api_key = "8380ab5954be5ca443fe95a5a05a2d61fa1e3d2fdffb88959bcd35da9607bcc8"  

csv_file = "cleaned_dataset.csv"
jsonl_file = "ncert_finetune.jsonl"

convert_csv_to_jsonl(csv_file, jsonl_file)
file_id = upload_dataset(api_key, jsonl_file)
model_id = fine_tune_model(api_key, file_id)

# Test the model
test_query = "What is Newton's First Law?"
response = chat_with_model(api_key, model_id, test_query)
print("Chatbot Response:", response)
