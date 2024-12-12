import os
import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
from dotenv import load_dotenv
import logging
import chardet
import argparse
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Fetch AI Proxy token
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    logging.error("AIPROXY_TOKEN not found in .env file. Please add it.")
    sys.exit(1)

headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {AIPROXY_TOKEN}'}

# Optimized function to load data
def load_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        data = pd.read_csv(file_path, encoding=encoding)
        logging.info(f"Data loaded with {encoding} encoding.")
        return data
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        sys.exit(1)

# Optimized outlier detection using vectorized IQR
def outlier_detection(data):
    numeric_data = data.select_dtypes(include=np.number)
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum()
    return {"outliers": outliers.to_dict()}

# Function for basic data analysis
def basic_analysis(data):
    summary = data.describe(include='all').to_dict()
    missing_values = data.isnull().sum().to_dict()
    column_info = data.dtypes.to_dict()
    return {"summary": summary, "missing_values": missing_values, "column_info": column_info}

# Function to save plots with better readability
def save_plot(fig, plot_name):
    plot_path = f"{plot_name}.png"
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Plot saved as {plot_path}")
    return plot_path

# Enhanced PCA Visualization
def generate_pca_plot(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()
    if numeric_data.shape[1] < 2:
        logging.warning("Insufficient numeric columns for PCA.")
        return None
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(StandardScaler().fit_transform(numeric_data))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], ax=ax, palette="viridis")
    ax.set_title("PCA Plot", fontsize=16)
    ax.set_xlabel("Principal Component 1", fontsize=14)
    ax.set_ylabel("Principal Component 2", fontsize=14)
    return save_plot(fig, "pca_plot")

# Optimized AI Story generation function
def get_ai_story(dataset_summary, dataset_info, visualizations):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    prompt = f"""
    Generate a narrative for the dataset analysis:
    1. **Data Description**
    2. **Analysis Methods Used**
    3. **Key Insights**
    4. **Implications for Action**
    5. **Visualizations and Interpretation**
    Dataset Summary: {dataset_summary}
    Dataset Info: {dataset_info}
    Visualizations: {visualizations}
    """
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        return "Error: Unable to generate narrative."

    return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No narrative generated.")

# Main analysis function
def analyze_and_generate_output(file_path):
    data = load_data(file_path)
    analysis = basic_analysis(data)
    outliers = outlier_detection(data)
    combined_analysis = {**analysis, **outliers}

    image_paths = {
        'pca_plot': generate_pca_plot(data),
    }

    data_info = {
        "filename": file_path,
        "summary": combined_analysis["summary"],
        "missing_values": combined_analysis["missing_values"],
        "outliers": combined_analysis["outliers"]
    }

    narrative = get_ai_story(data_info["summary"], data_info["missing_values"], image_paths)
    if not narrative:
        narrative = "Error: Narrative generation failed."
    save_readme(f"Dataset Analysis: {narrative}")
    return narrative, image_paths

# Main entry point
def main():
    if len(sys.argv) != 2:
        logging.error("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_and_generate_output(file_path)

if __name__ == "__main__":
    main()
