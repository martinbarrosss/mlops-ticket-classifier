# 🚀 MLOps Customer Support Ticket Classifier

An end-to-end Machine Learning Operations (MLOps) pipeline that fine-tunes a Large Language Model (DistilBERT) to automatically classify customer support tickets into actionable intents. The model is deployed as a production-ready REST API using FastAPI and Docker.

## 🧠 Project Overview
Customer support teams spend countless hours manually triaging tickets. This project solves that problem by using Natural Language Processing (NLP) to read the user's complaint and predict the correct category (e.g., `refund_request`, `cancel_order`, `recover_password`) with a high degree of confidence.

### Key Features:
* **Custom Fine-Tuning:** Trained a Hugging Face `distilbert-base-uncased` model on a custom customer support dataset.
* **Dynamic Configuration:** Implemented robust inference scripts that dynamically reconstruct missing Hugging Face config architectures on the fly.
* **REST API:** Built a high-performance, asynchronous API using FastAPI.
* **Containerized Deployment:** Fully dockerized the application to eliminate dependency conflicts across environments.

## 🛠️ Tech Stack
* **Machine Learning:** PyTorch, Hugging Face (Transformers, Datasets, Evaluate)
* **Data Processing:** Pandas, Scikit-learn
* **Backend:** FastAPI, Uvicorn, Pydantic
* **DevOps:** Docker, Git

## 📂 Project Structure
```text
mlops-ticket-classifier/
├── app/
│   └── main.py                 # FastAPI server and endpoints
├── data/
│   └── processed/              # Label mappings and split datasets
├── models/
│   └── ticket_classifier_final/# Fine-tuned safetensors and configs
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory Data Analysis
├── src/
│   ├── data_prep.py            # Data cleaning and encoding script
│   ├── train.py                # DistilBERT fine-tuning script
│   └── predict.py              # Local inference script
├── Dockerfile                  # Container instructions
├── requirements.txt            # Production dependencies
└── README.md
```

## 🚀 Future Improvements
While the current pipeline is fully functional, next steps for scaling include:
* **CI/CD Pipeline:** Implementing GitHub Actions to automatically run tests and build the Docker image on every push.
* **Cloud Deployment:** Deploying the Docker container to AWS ECS or Google Cloud Run for public access.
* **Model Monitoring:** Integrating tools like MLflow or Prometheus to track data drift and model degradation over time.

## 📬 Author
**Martín Barros Iglesias**
* **LinkedIn:** www.linkedin.com/in/martín-barros
* **Email:** martin.barros.personal@gmail.com