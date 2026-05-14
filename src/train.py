import json
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate

def train_model():
    print("1. Loading processed data and label mappings...")
    dataset = load_from_disk("data/processed/hf_dataset")

    # Load mappings to tell the model exactly what each number means
    with open("data/processed/label_mapping.json", "r") as f:
        mappings = json.load(f)
        label2id = mappings["label2id"]
        # JSON saves keys as strings, we need them as integers for the model
        id2label = {int(k): v for k, v in mappings["id2label"].items()} 

    print("2. Loading Tokenizer and Model (DistilBERT)...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize the model with our specific number of categories
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    print("3. Tokenizing the dataset (Converting text to numbers)...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    print("4. Setting up Training Arguments...")
    # This dictates how the model learns. We will train for 3 epochs (3 full passes over the data).
    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    # Function to calculate accuracy during training
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    print("5. Initializing Trainer and starting fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("6. Saving the final production model...")
    trainer.save_model("models/ticket_classifier_final")
    tokenizer.save_pretrained("models/ticket_classifier_final")

    print("Training complete. Model successfully saved.")

if __name__ == "__main__":
    train_model()