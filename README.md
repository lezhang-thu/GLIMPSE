# GLIMPSE: Global LLM-Driven Template Merging for Unsupervised Log Parsing

This is the official repo for our COMPSAC 2026 paper: _**GLIMPSE: Global LLM-Driven Template Merging for Unsupervised Log Parsing**_

# 1. Quick Start
### Step 1: Environment
Run the following command to install the dependencies
```bash
conda create -n glimpse python=3.8.19
conda activate glimpse

cd GLIMPSE
pip install -r requirements.txt
```

### Step 2: Add openai API key and base url
Add your openai API key and base url by updating `./GLIMPSE/config.py` file
```python
LLM_BASE_MAPPING = {
    "DeepSeek-V3.2": [
        "deepseek-chat",
        "https://api.deepseek.com",
        ”API_KEY“,
    ]
}
```

### Step 3: Run with example Apache dataset
```commandline
bash run.sh Apache
```

# 2. Run with more datasets
To experiment with more datasets, please follow the below steps:

### Step 1: Download Loghub-2.0
You can download the other datasets in LogHub2.0 from this [link](https://zenodo.org/records/8275861) and move them to `./dataset` just like the Apache dataset.

Please perform the following preprocessing step to canonicalize logs for **Apache**, **HPC**, **Hadoop**, and **Spark**, using the provided canonical template files (e.g., `Apache_full.log_templates.csv`):

```commandline
cd datasets
# Example for Hadoop
python template_2_structure.py Hadoop_full.log_templates.csv Hadoop_full.log_structured.csv
```

### Step 2: Run with all datasets
You can run the following command to evaluate on all datasets
```commandline
bash run.sh all
```


# 3. GLIMPSE-Parallel
You can also run an efficient version GLIMPSE-Parallel with the following steps:

### Step 1: Run with example Apache dataset
```commandline
bash run-parallel.sh Apache
```

### Step 2: Run with Loghub-2.0
After you download the Loghub-2.0 datasets, you can use the following scripts to evaluate on all datasets
```commandline
bash run-parallel.sh all
```
# 4. Evaluation Details
For readers seeking the detailed evaluation metrics referenced in our paper, please find the supplementary materials in the [`paper-materials`](./paper-materials) directory:

* **Cost Analysis & Token Consumption**: Detailed pricing and API token usage statistics across different parsers are available in [**Table VI**](./paper-materials/TABLE%20VI-cost%20analysis%20and%20toke...). *(Note: Please ensure this link matches your exact file name)*
* **Runtime Efficiency**: The comprehensive runtime comparison on the Loghub-2.0 dataset can be found in [**Figure 3**](./paper-materials/Fig.3-Runtime%20Comparison.jpg).
