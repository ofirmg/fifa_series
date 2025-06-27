# Football Insights from FIFA Data - Medium series

This repo contains the code base for the <a href="https://medium.com/@ofirmagdaci">"Football Insights from FIFA
Data"</a> articles series by Ofir Magdaci at medium.com.<br>
The project consists of three posts:
1. <a href="https://medium.com/@ofirmagdaci/football-insights-from-fifa-data-what-comes-and-goes-with-age-2c4636bc99d1">
   What Comes and Goes with Age</a> - a deep dive into how football attributes evolve with player age.
2. <a href="https://medium.com/@ofirmagdaci/football-insights-from-fifa-data-player-valuation-55b1b748e05d">Player
   Valuation</a> - estimating player market value based on performance and age.
3. <a href="[https://medium.com/@ofirmagdaci/](https://medium.com/@ofirmagdaci/football-insights-from-fifa-data-predicting-the-future-59891408e8dd)">Predicting the Future</a> - modeling future growth and identifying
   rising stars.

## ⚠️ Notes & Disclaimers
1. This code is an experimental playground, intended for self-exploration rather than production use. I haven’t focused on optimization, formatting, or full documentation. While I’ve done some testing and verification, bugs and inefficiencies likely exist. 
2. The FIFA dataset is not publicly available and was assembled manually from third-party sources.
3. The code in this repository mostly cover Part 3 (Predicting the Future).

## 🧠 The Stat Predict Framework
A general-purpose framework for fitting machine learning models to FIFA player attributes. Controlled via centralized configs and extensible to support new labels, data sources, and model types.
Currently set up for binary classification, but can be adapted for regression or multi-class problems.

![data_flow.png](artifacts/figures/data_flow.png)
Figure 1. A schematic of the full modeling flow. Raw data is enriched both tabularly and temporally, then filtered and passed into the final classifier.

## 📦 Project Structure
stat_predict/<br>
├── dataset/&nbsp;&nbsp;&nbsp;&nbsp;# Feature engineering & dataset preparation<br>
├── eval/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Model evaluation and plotting utilities<br>
├── models/&nbsp;&nbsp;&nbsp;&nbsp;# Modeling logic (baselines, deep learning, time-series)<br>
├── static/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Configs, constants, and bin definitions<br>
├── data/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Raw data (not included in repo)<br>
├── artifacts/ &nbsp;&nbsp;&nbsp;# Output: models, plots, reports<br>
└── main.py &nbsp;&nbsp;&nbsp;# Entry point for running experiments<br>

## Key Modules
- 📁 dataset/
    - features_utils.py: Constructs and manages feature groups.
    - label.py: Label encoding logic for classification.
    - sp_dataset.py: Main pipeline for loading, cleaning, enriching, and slicing data into time-series windows.
    - utils.py: Utility functions for feature typing, scaling, and diagnostics.
- 🤖 models/
    - model.py: MLStatPredict class – the core training/evaluation engine for all models.
    - baselines.py: Baseline models - Logistic Regression, XGBoost, CatBoost and TabNet.
    - career_phase.py: Time-series clustering of player careers and phase-based modeling.
    - dl.py: Deep learning architectures (DLStatPredict) and training logic (PyTorch), adjusting the broader MLStatPredict interface.
    - experiment.py: Wraps the full modeling process into a one-call experiment flow.
- 📊 eval/
    - evaluation.py: Main entry point for evaluating a model’s predictions against ground truth.
    - utils.py: Curated player lists and utils functions for evaluation.
- ⚙️ static/
    - config.py: Global constants for feature selection, binning, training parameters and I/O paths.
    - utils.py: Canonical definitions for columns, positions, and FIFA metadata.

## 🤖 Supported Models
### ✅ Baselines
Each supports grid/random search, feature importance, classes-weights balancing, model saving/loading
	•	Logistic Regression (LRStatPredict)
	•	XGBoost (XGBStatPredict)
	•	CatBoost (CatBoostStatPredict)
	•	TabNet (TabNetStatPredict)

### 🔥 Deep Learning (DLStatPredict)
DLStatPredict is a general wrapper which extends the MLStatPredict for PyTorch models: 
- Implementing a full PyTorch training loop, early stopping, appropriate loss (CrossEntropyLoss) and adds convergence plots. <br>
- In addition, it supports both 1D (flat) and 2D (time-series) like inputs. 

The code supports any PyTorch nn.module with binary output as its classifier. It offers two ready architectures to work with:
- Multi-Layer-Perceptron (MLP)
- HistEmbedNet

### Possible Experimentation
All models can be further experimented across multiple dimensions:
- Data horizon: Max number of years to use as history
- Feature selection process - such as correlation type/ threshold, number of features to select and additional filtering.
- Phase model version to use
- Additional models
- Models hyper-parameters

### PhaseModel
The model (class SplitCareerPhaseModel) trains separate phase models per slice (e.g. by rating or age group). Evaluates across slices and unifies results.
Within each slice, it uses TimeSeriesKMeans + KNN to cluster career paths and predict player development.<br>
**Possible experimentation:**
- Slicing attributes, such as age, position and rating.
- Clustering attributes, default is the player rating.
- Data horizon: I used 3 years.
- Clustering parameters: Max number of clusters per slice, and k (num of neighbors) to use for the KNN model. 
- Time series manipulation (e.g., diff, pct_change)
- Missing data imputation strategy

## 📊 Data Summary
The dataset includes:
• 53,111 unique players
• 180,021 player-year records
• FIFA game editions: 2015–2024 (2025 only used for labels)
• Snapshot structure: each row is a static record of a player’s attributes in a single edition (no intra-year dynamics)

## 🔧 Setup & Requirements

Install the required Python packages:

```aiignore
pip install -r requirements.txt
```

## 📈 Sample Use

```aiignore
python main.py
```

## 📄 License

As described in the license file — feel free to use or adapt, but no warranties provided.

## Contacts and communication:

- 🌐 <a href="www.magdaci.com">Personal website</a>
- 🐦 <a href="https://twitter.com/Magdaci">Twitter</a>
- <a href="https://medium.com/@ofirmagdaci">Medium Blog</a>
