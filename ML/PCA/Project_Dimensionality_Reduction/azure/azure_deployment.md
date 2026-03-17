# Azure ML Deployment for PCA Dimensionality Reduction

## Overview

This document describes deploying the PCA dimensionality reduction model to Azure Machine Learning for scalable inference.

## Prerequisites

- Azure subscription
- Azure ML workspace
- Azure Container Registry (ACR)
- Azure Storage Account
- Docker installed locally
- Azure CLI installed

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client App   │────▶│  Azure ML Endpoint│◀────│  PCA Model     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Azure Storage   │
                    │  (Model + Data)  │
                    └──────────────────┘
```

## Deployment Steps

### 1. Prepare Environment

```bash
# Login to Azure
az login
az account set --subscription <subscription-id>

# Create resource group
az group create --name pca-rg --location eastus

# Create Azure ML workspace
az ml workspace create --workspace-name pca-workspace --resource-group pca-rg
```

### 2. Create Scoring Script

Create `score.py` for real-time inference:

```python
import json
import numpy as np
import joblib
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('pca_model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        transformed = model.transform(data)
        return json.dumps({
            'transformed_data': transformed.tolist(),
            'explained_variance': model.explained_variance_ratio_.tolist()
        })
    except Exception as e:
        return json.dumps({'error': str(e)})
```

### 3. Create Environment YAML

Create `environment.yml`:

```yaml
name: pca-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.8
  - pip
  - pip:
    - numpy
    - scikit-learn
    - pandas
    - azureml-core
    - azureml-defaults
```

### 4. Register Model

```python
from azureml.core import Workspace, Model
import os

ws = Workspace.from_config()
model = Model.register(
    workspace=ws,
    model_name='pca_model',
    model_path='models/pca_model.pkl',
    description='PCA dimensionality reduction model'
)
```

### 5. Create Inference Config

```python
from azureml.core import Environment
from azureml.core.model import InferenceConfig

env = Environment.from_conda_specification(
    name='pca-env',
    file_path='environment.yml'
)

inference_config = InferenceConfig(
    entry_script='score.py',
    environment=env
)
```

### 6. Deploy to AKS

```python
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import AksWebservice

# Create AKS cluster
aks_compute = AksCompute.provisioning_configuration(
    agent_count=2,
    vm_size='Standard_DS2_v2'
)
aks_target = ComputeTarget.create(ws, 'pca-aks', aks_compute)

# Deploy
deployment_config = AksWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    enable_app_insights=True
)

service = Model.deploy(
    workspace=ws,
    name='pca-service',
    models=[model],
    inference_config=inference_config,
    deployment_target=aks_target,
    deployment_config=deployment_config
)
service.wait_for_deployment(show_output=True)
```

### 7. Test Endpoint

```python
import requests
import json

endpoint = service.scoring_uri
api_key = service.get_keys()[0]

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

test_data = {
    'data': [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]
}

response = requests.post(endpoint, json=test_data, headers=headers)
print(response.json())
```

## CI/CD Pipeline

### Azure DevOps Pipeline

Create `azure-pipelines.yml`:

```yaml
trigger:
  branches:
    include:
      - main

resources:
  - repo: self

stages:
  - stage: Train
    jobs:
      - job: Training
        pool:
          vmImage: 'ubuntu-latest'
        steps:
          - task: PythonScript@0
            inputs:
              scriptSource: inline
              script: |
                python -m src.pipelines.training_pipeline

  - stage: Register
    jobs:
      - job: Register
        pool:
          vmImage: 'ubuntu-latest'
        steps:
          - task: AzureMLModelRegister@1
            inputs:
              azureSubscription: 'AzureML'
              ModelName: 'pca_model'
              ModelPath: 'models/pca_model.pkl'

  - stage: Deploy
    jobs:
      - deployment: Deploy
        pool:
          vmImage: 'ubuntu-latest'
        environment: 'pca-prod'
        strategy:
          runOnce:
            deploy:
              steps:
                - task: AzureMLWebserviceDeploy@1
                  inputs:
                    azureSubscription: 'AzureML'
                    webserviceName: 'pca-service'
```

## Monitoring

### Application Insights

Enable monitoring in deployment config:

```python
deployment_config = AksWebservice.deploy_configuration(
    enable_app_insights=True,
    collect_model_data=True
)
```

### Metrics to Track

- Request latency (p50, p95, p99)
- Request count
- Model transformation time
- Error rate
- CPU/Memory usage

## Scaling

### Auto-scaling

```python
from azureml.core.webservice import AksWebservice

service.update(
    autoscale_enabled=True,
    min_replicas=1,
    max_replicas=10,
    target_utilization_percent=70
)
```

## Security

### Authentication

```python
deployment_config = AksWebservice.deploy_configuration(
    auth_enabled=True,
    token_auth_enabled=True
)
```

### SSL/TLS

```python
deployment_config = AksWebservice.deploy_configuration(
    ssl_enabled=True,
    ssl_cert_pem_file="cert.pem",
    ssl_key_pem_file="key.pem",
    ssl_cname="pca.example.com"
)
```
