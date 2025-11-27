# Respiratory Sound Classification System

![Python](https://img.shields.io/badge/Python-3.9%252B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)

A comprehensive machine learning system for classifying respiratory
sounds into four categories: **Normal**, **Crackle**, **Wheeze**, and
**Both**. This project includes a complete ML pipeline, web interface,
API, and deployment infrastructure.

##  Video Demonstration

[![Link
Demo](https://www.canva.com/design/DAG57fHYMTg/apk8YydAacpkm9sP1ZJH9w/edit?utm_content=DAG57fHYMTg&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)]

> Replace with your actual YouTube link.

##  Live Demo

**Web Application:** 


##  Project Overview

This system provides: - ML Classification Model for respiratory sounds -
Web UI for monitoring and interactions - REST API for predictions -
Retraining Pipeline for model improvement - Load Testing with
performance metrics - Cloud Deployment with Docker containers

##  System Architecture

    User Interface (Streamlit) → FastAPI → ML Model → Database
           ↑                            ↑          ↑
        Monitoring                  Predictions  Retraining
           ↓                            ↓          ↓
       Visualizations              Load Testing  Data Pipeline

##  Repository Structure

    respiratory-sound-classification/
    │
    ├──  notebook/
    │   └── respiratory_sound_analysis.ipynb  # Complete ML pipeline
    │
    ├──  src/
    │   ├── preprocessing.py    # Audio preprocessing utilities
    │   ├── model.py            # Model architecture and training
    │   ├── prediction.py       # Prediction functions
    │   └── api.py              # FastAPI application
    │
    ├──  webapp/
    │   └── app.py              # Streamlit dashboard
    │
    ├──  data/
    │   ├── train/              # Training data
    │   ├── test/               # Testing data
    │   └── processed/          # Processed datasets
    │
    ├──  models/
    │   ├── best_model.h5       # Trained TensorFlow model
    │   └── model_checkpoints/  # Training checkpoints
    │
    ├──  deployment/
    │   ├── Dockerfile
    │   ├── docker-compose.yml
    │   ├── nginx.conf
    │   └── locustfile.py       # Load testing configuration
    │
    ├──  requirements.txt     # Python dependencies
    └── README.md

##  Installation & Setup

### Prerequisites

-   Python 3.9 or higher
-   Docker and Docker Compose
-   Git

### Method 1: Local Installation

Clone the repository:

``` bash
git https://github.com/rodwol/classification_model.git
cd classification_model
```

Create virtual environment:

``` bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Download the dataset:


Run the notebook:

``` bash
jupyter notebook https://colab.research.google.com/drive/1MEYMdL18r06-QNOZ8rSNWLS9_wi7Aa70?usp=sharing
```

### Method 2: Docker Deployment

``` bash
docker-compose up -d --build
```

Access apps: - http://localhost:8501 - http://localhost:8000/docs -
http://localhost:8089

## Model Performance

  Metric      Value
  ----------- -------
  Accuracy    92.3%
  F1-Score    0.89
  Precision   0.91
  Recall      0.88
  Loss        0.215

Confusion Matrix:

              Predicted
             N  C  W  B
    Actual N 45  2  1  0
           C  3 38  4  1  
           W  1  2 42  2
           B  0  1  3 39

##  Features

-   Model Prediction
-   Data Visualizations
-   Bulk Data Upload
-   Model Retraining
-   System Monitoring

##  API Endpoints

-   POST /predict
-   POST /batch-predict
-   GET /model-info
-   GET /health
-   GET /metrics
-   GET /uptime
-   POST /retrain
-   GET /training-status

##  License

MIT

## Acknowledgments

ICBHI 2017 Dataset, TensorFlow, FastAPI, Streamlit, Docker
