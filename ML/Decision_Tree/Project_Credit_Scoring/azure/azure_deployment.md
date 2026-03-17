# Azure Deployment Guide - Credit Scoring Model

This guide provides step-by-step instructions for deploying the Credit Scoring Decision Tree model on Microsoft Azure.

---

## Prerequisites

- Azure Account (https://azure.microsoft.com)
- Azure CLI installed
- Python 3.8+
- Docker (for containerization)

---

## Step 1: Create Azure Resource Group

```bash
# Login to Azure
az login

# Set your subscription
az account set --subscription "Your-Subscription-Name"

# Create resource group
az group create --name credit-scoring-rg --location eastus
```

---

## Step 2: Create Azure Machine Learning Workspace

```bash
# Create AML workspace
az ml workspace create \
  --workspace-name credit-scoring-ws \
  --resource-group credit-scoring-rg \
  --location eastus
```

---

## Step 3: Create Compute Target

```bash
# Create compute cluster
az ml compute create \
  --name cpu-cluster \
  --workspace-name credit-scoring-ws \
  --resource-group credit-scoring-rg \
  --type AmlCompute \
  --size STANDARD_DS2_V2 \
  --min-instances 0 \
  --max-instances 2
```

---

## Step 4: Upload Data to Azure Blob Storage

```bash
# Create storage account
az storage account create \
  --name creditstorage$(guid) \
  --resource-group credit-scoring-rg \
  --location eastus \
  --sku Standard_LRS

# Get storage key
az storage account keys list \
  --account-name creditstorage$(guid) \
  --resource-group credit-scoring-rg

# Create container
az storage container create \
  --name data \
  --account-name creditstorage$(guid) \
  --account-key <your-key>

# Upload data
az storage blob upload \
  --container-name data \
  --name credit_data.csv \
  --file data/credit_data.csv \
  --account-name creditstorage$(guid) \
  --account-key <your-key>
```

---

## Step 5: Create Training Pipeline

Create `azure/pipeline.yml`:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline
name: credit_scoring_pipeline
display_name: Credit Scoring Training Pipeline
experiment_name: credit_decision_tree

settings:
  default_compute: cpu-cluster
  default_datastore: workspaceblobstore

inputs:
  data:
    type: uri_file
    path: azureml://datastores/workspaceblobstore/paths/data/credit_data.csv

outputs:
  model_output:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/model

jobs:
  training_job:
    type: command
    command: >-
      python src/pipelines/training_pipeline.py
    environment: azureml:AzureML-sklearn-1.0-ubuntu20.04@latest
    compute: cpu-cluster
    inputs:
      data: ${{inputs.data}}
    outputs:
      model_output: ${{outputs.model_output}}
```

**Submit the pipeline:**

```bash
az ml job create \
  --file azure/pipeline.yml \
  --workspace-name credit-scoring-ws \
  --resource-group credit-scoring-rg
```

---

## Step 6: Register Model

```bash
# After training completes, register the model
az ml model register \
  --name credit-scoring-model \
  --version 1 \
  --path azureml://model-output/credit_decision_tree.joblib \
  --workspace-name credit-scoring-ws \
  --resource-group credit-scoring-rg
```

---

## Step 7: Create Scoring Script

Create `azure/score.py`:

```python
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('credit-scoring-model')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        data_array = np.array(data)
        prediction = model.predict(data_array)
        return json.dumps({"prediction": prediction.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

---

## Step 8: Create Inference Configuration

Create `azure/inference_config.yml`:

```yaml
name: credit-scoring-inference
entry_script: score.py
runtime: python
conda_file: requirements.txt

columns:
  - age
  - income
  - credit_score
  - employment_years
  - loan_amount
```

---

## Step 9: Deploy to Azure Container Instance (ACI)

```bash
az ml model deploy \
  --name credit-scoring-service \
  --model credit-scoring-model:1 \
  --inference-config inference_config.yml \
  --workspace-name credit-scoring-ws \
  --resource-group credit-scoring-rg \
  --deployment-type ACI \
  --cpu-core 1 \
  --memory-gb 1
```

---

## Step 10: Test the Deployment

```bash
# Get the scoring URI
az ml service show \
  --name credit-scoring-service \
  --workspace-name credit-scoring-ws \
  --resource-group credit-scoring-rg

# Test with sample data
curl -X POST <scoring-uri> \
  -H 'Content-Type: application/json' \
  -d '[[30, 50000, 650, 5, 10000]]'
```

---

## Azure Visualizations

### Azure ML Studio

1. **Navigate to Azure ML Studio**: https://ml.azure.com
2. **View Experiments**: Track all training runs
3. **Compare Runs**: See metrics across different configurations
4. **View Charts**: Accuracy, Precision, Recall over time

### Azure Dashboard

1. **Create Dashboard** in Azure Portal
2. **Add Metrics Widget**: Model accuracy, prediction latency
3. **Add Application Insights**: Request rates, failures
4. **Add Log Analytics**: Query custom logs

### Power BI Integration

```python
# Export predictions to Power BI
import pandas as pd
from azure.storage.blob import BlobServiceClient

# Save predictions
predictions_df.to_csv('predictions.csv', index=False)

# Upload to Azure Blob
blob_service_client = BlobServiceClient.from_connection_string("<conn-string>")
blob_client = blob_service_client.get_blob_client(container="predictions", blob="predictions.csv")
blob_client.upload_blob(predictions_df)
```

Then connect Power BI to Azure Blob Storage for visualization.

---

## Monitoring & Alerts

```bash
# Set up Application Insights
az monitor app-insights component create \
  --app credit-scoring-insights \
  --location eastus \
  --resource-group credit-scoring-rg

# Create alert rule
az monitor alert create \
  --name high-latency-alert \
  --resource-group credit-scoring-rg \
  --condition "requests | where duration > 1000" \
  --action-group email-admin
```

---

## Cleanup

```bash
# Delete the deployment
az ml service delete \
  --name credit-scoring-service \
  --workspace-name credit-scoring-ws \
  --resource-group credit-scoring-rg

# Delete resource group (removes everything)
az group delete --name credit-scoring-rg --yes