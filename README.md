# LogAD - Anomaly Detection in Log Data

This project focuses on analyzing and implementing anomaly detection techniques for log data, particularly the HDFS (Hadoop Distributed File System) logs.

## Project Structure

```
LogAD/
├── data/                # Raw and processed datasets
│   └── HDFS/            # HDFS dataset 
├── notebooks/           # Jupyter notebooks
│   └── hdfs_analysis.ipynb  # Analysis of HDFS dataset
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup

1. Create and activate the conda environment:
```bash
conda create -n logad python=3.10 -y
conda activate logad
conda install -y numpy=1.23.5 pandas=1.5.3 matplotlib=3.7.1 seaborn=0.12.2 jupyter scikit-learn=1.2.2 tqdm
```

2. Download the HDFS dataset:
```bash
# Download and extract the dataset
cd data
Invoke-WebRequest -Uri "https://zenodo.org/records/8196385/files/HDFS%5Fv1.zip?download=1" -OutFile "HDFS_v1.zip"
Expand-Archive -Path "HDFS_v1.zip" -DestinationPath "HDFS" -Force
```

## Usage

1. Start Jupyter notebook:
```bash
jupyter notebook
```

2. Open and run the analysis notebook at `notebooks/hdfs_analysis.ipynb`

## Dataset

The HDFS dataset contains log data from the Hadoop Distributed File System and includes:

- Raw log file (HDFS.log)
- Preprocessed data:
  - Log templates
  - Anomaly labels
  - Event traces
  - Pre-processed features

This dataset is widely used for benchmarking log-based anomaly detection techniques.
