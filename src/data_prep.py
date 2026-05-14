import os
import json
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def prepare_data():
    print("1. Loading dataset from Hugging Face...")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = dataset['train'].to_pandas()

    # Keep only necessary columns and rename them for standard HF conventions
    df = df[['instruction', 'intent']]
    df = df.rename(columns={'instruction': 'text', 'intent': 'label_text'})

    print("2. Encoding labels (Text to Integer)...")
    unique_labels = df['label_text'].unique().tolist()
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    # Create the numeric label column
    df['label'] = df['label_text'].map(label2id)

    print("3. Splitting data into Train, Validation, and Test sets...")
    # 80% Train, 20% for Validation/Test combined. 
    # stratify=df['label'] ensures equal class distribution in all splits.
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Split the 20% into 10% Validation and 10% Test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    print(f"Dataset split: Train({len(train_df)}), Validation({len(val_df)}), Test({len(test_df)})")

    print("4. Converting back to Hugging Face DatasetDict format...")
    hf_dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'validation': Dataset.from_pandas(val_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df, preserve_index=False)
    })

    print("5. Saving processed datasets and label mappings to disk...")
    # Create the processed data directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Save the HF dataset to disk
    hf_dataset.save_to_disk("data/processed/hf_dataset")

    # Save the label mapping for future inference
    with open("data/processed/label_mapping.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)

    print("Data preparation complete")

if __name__ == "__main__":
    prepare_data()