from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import load_inference_components, predict_ticket

# 1. Define the application
app = FastAPI(
    title="Support Ticket Classifier API",
    description="MLOps API for classifying customer support tickets using DistilBERT.",
    version="1.0.0"
)

# Global variables to keep the model in memory
tokenizer = None
model = None
id2label = None

# 2. Define data schemas (Input and Output)
class TicketRequest(BaseModel):
    text: str

class TicketResponse(BaseModel):
    category: str
    confidence: float

# 3. Load the model on server startup
@app.on_event("startup")
async def startup_event():
    global tokenizer, model, id2label
    print("Starting server: Loading AI model into memory...")
    tokenizer, model, id2label = load_inference_components()
    print("Model loaded and ready to receive requests!")

# 4. Create the main prediction endpoint
@app.post("/predict", response_model=TicketResponse)
async def predict_endpoint(request: TicketRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Ticket text cannot be empty.")
    
    # Use the function imported from predict.py
    category, confidence = predict_ticket(request.text, tokenizer, model, id2label)
    
    return TicketResponse(category=category, confidence=confidence)

# 5. Health check endpoint to verify API is running
@app.get("/")
async def root():
    return {"message": "API is operational. Visit /docs to test it."}