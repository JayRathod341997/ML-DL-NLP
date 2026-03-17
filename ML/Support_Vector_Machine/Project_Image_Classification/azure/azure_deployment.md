# Azure Deployment Guide

## Overview
This guide covers deploying the SVM Image Classification model to Azure cloud services.

## Prerequisites
- Azure subscription
- Azure CLI installed
- Docker installed (for containerized deployment)
- Trained model files (model.pkl, scaler.pkl)

## Deployment Options

### Option 1: Azure ML Studio

#### Step 1: Prepare Environment
```bash
# Login to Azure
az login

# Create resource group
az group create --name svm-rg --location eastus

# Create Azure ML workspace
az ml workspace create --name svm-workspace --resource-group svm-rg
```

#### Step 2: Register Model
```python
from azureml.core import Workspace, Model
import os

ws = Workspace.from_config()
model = Model.register(
    workspace=ws,
    model_name="svm-classifier",
    model_path="models/svm_model.pkl",
    description="SVM Image Classifier"
)
```

#### Step 3: Create Scoring Script
Create `score.py`:
```python
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model, scaler
    model_path = Model.get_model_path("svm_model.pkl")
    scaler_path = Model.get_model_path("scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data))
        data_scaled = scaler.transform(data)
        predictions = model.predict(data_scaled)
        return json.dumps(predictions.tolist())
    except Exception as e:
        return json.dumps({"error": str(e)})
```

#### Step 4: Create Environment
Create `env.yml`:
```yaml
name: svm-env
dependencies:
  - python=3.8
  - scikit-learn
  - numpy
  - pandas
```

#### Step 5: Deploy
```bash
# Deploy to Azure Container Instance
az ml service create \
    --name svm-service \
    --workspace svm-workspace \
    --model svm_model.pkl \
    --score-file score.py \
    --environment env.yml \
    --cpu-cores 2 \
    --memory-gb 4
```

### Option 2: Azure Kubernetes Service (AKS)

#### Step 1: Create AKS Cluster
```bash
az aks create \
    --resource-group svm-rg \
    --name svm-cluster \
    --node-count 3 \
    --enable-managed-identity
```

#### Step 2: Attach ACR
```bash
az aks update \
    --name svm-cluster \
    --resource-group svm-rg \
    --attach-acr svmregistry
```

#### Step 3: Deploy Service
```python
from azureml.core.webservice import AksWebservice, Webservice

aks_config = AksWebservice.deploy_configuration(
    cpu_cores=2,
    memory_gb=4,
    autoscale_enabled=True,
    min_replicas=1,
    max_replicas=3
)

service = Webservice.deploy_from_model(
    workspace=ws,
    name="svm-aks-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=aks_config,
    deployment_target="Kubernetes"
)
service.wait_for_deployment(show_output=True)
```

### Option 3: Azure Functions (Serverless)

#### Step 1: Create Function App
```bash
az functionapp create \
    --resource-group svm-rg \
    --consumption-plan-location eastus \
    --name svm-function \
    --storage-account svmstorage
```

#### Step 2: Deploy Code
```bash
# Package and deploy
func azure functionapp publish svm-function
```

## Monitoring

### Application Insights
```python
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(
    connection_string="InstrumentationKey=<your-key>"
))
```

### Metrics to Track
- Request latency (p50, p95, p99)
- Prediction accuracy
- Error rate
- Model confidence distribution
- Input data drift

## Cost Optimization

1. **Use Auto-scaling**: Configure AKS autoscaling
2. **Batch Predictions**: Process multiple images together
3. **Spot Instances**: Use preemptible VMs for development
4. **Azure ML Endpoints**: Use managed endpoints with auto-scale

## Security

1. **Enable HTTPS**: Use TLS for all connections
2. **Key Vault**: Store secrets in Azure Key Vault
3. **VNet**: Deploy in virtual network for isolation
4. **Role-based Access**: Use managed identities

## Troubleshooting

### Issue: Service Not Starting
- Check logs: `az ml service logs <service-name>`
- Verify model file exists
- Check scoring script for errors

### Issue: High Latency
- Check instance size
- Review batch size
- Enable caching for repeated queries

### Issue: Out of Memory
- Reduce model complexity
- Use smaller instance
- Implement request queuing
