# AI-Powered Size Chart Generator for Apparel Sellers

## Description
This AI system generates accurate size charts for apparel sellers based on user body measurements and purchase history. It clusters similar body types, analyzes past purchases, and provides confidence scores for each size recommendation.

## Features
- Clustering based on user measurements (height, weight, chest, waist, hips).
- Confidence score for size recommendations.
- Adaptive size chart that updates with new data.

## Installation Steps
To install and run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AI-Powered-Size-Chart-Generator.git
   cd AI-Powered-Size-Chart-Generator
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:
   ```bash
   python main.py
   ```

## Datasets
- `datasets/body_measurements_dataset.csv`: Main dataset with body measurements.
- `datasets/generated_body_measurements_dataset.csv`: Supplementary dataset for analysis.

## How It Works
1. Preprocesses the data (converts height to inches, fills missing values).
2. Clusters data using KMeans based on selected features (e.g., height, weight, bust).
3. Generates size charts for categories (tops, bottoms, dresses) based on clusters.

## Made by-Kedar Singh
