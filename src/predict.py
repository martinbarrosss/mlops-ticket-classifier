import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

def load_inference_components():

    # 1. Load mappings from our JSON
    with open("data/processed/label_mapping.json", "r") as f:
        mappings = json.load(f)
        id2label = {int(k): v for k, v in mappings["id2label"].items()}
        label2id = mappings["label2id"]

    model_path = "models/ticket_classifier_final"
    
    # 2. Fetch the tokenizer directly from the Hub (since local files are missing)
    print("Loading base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 3. Reconstruct the missing configuration dynamically!
    print("Reconstructing missing configuration...")
    config = AutoConfig.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=len(id2label), 
        id2label=id2label, 
        label2id=label2id
    )

    # 4. Load the local model weights injecting our reconstructed config
    print("Loading model weights...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    
    return tokenizer, model, id2label

def predict_ticket(text, tokenizer, model, id2label):
    # 1. Convert text to numbers (tokens) suitable for PyTorch ("pt")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # 2. Perform inference without calculating gradients (saves memory)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 3. Calculate probabilities using softmax and get the highest score
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class_id].item()
    
    # 4. Return the human-readable category and the confidence score
    return id2label[predicted_class_id], confidence

if __name__ == "__main__":
    print("Loading model for inference...")
    tokenizer, model, id2label = load_inference_components()
    
    # Nuevos tickets alineados con el dataset de entrenamiento (E-commerce / Cuentas)
    test_tickets = [
        "I forgot my password and cannot access my account, please help.",
        "I want to cancel my order #12345, I bought the wrong item by mistake.",
        "Where is my package? The tracking link says delivered but I have received nothing.",
        "The t-shirt I received is torn, I want my money back immediately."
    ]
    
    print("\n--- RUNNING PREDICTIONS ---")
    for ticket in test_tickets:
        category, confidence = predict_ticket(ticket, tokenizer, model, id2label)
        print(f"\nTicket: '{ticket}'")
        print(f"Predicted Category: {category} (Confidence: {confidence:.2%})")