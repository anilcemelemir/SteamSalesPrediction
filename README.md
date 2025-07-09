# 🎮 SteamSalesPrediction

**SteamSalesPrediction** is a machine learning project designed to predict the success of PC games on the Steam platform based on historical and metadata features. The project analyzes patterns in game attributes and uses regression models to estimate potential sales performance.

## 🚀 Features

- Cleaning and preprocessing of Steam game data  
- Feature engineering from genres, release dates, developer info, tags, etc.  
- Label creation using success metrics (sales tiers or proxy indicators)  
- Supervised learning with:
  - Linear Regression
  - Random Forest
  - XGBoost  
- Visual evaluation and model comparison (scatter plots, error metrics)

## 🧰 Tech Stack

- Python 3.x  
- pandas, numpy  
- scikit-learn  
- xgboost  
- matplotlib, seaborn  
- requests (for API-based updates or external features)

## 📦 Installation

pip install -r requirements.txt
python main.py

Make sure that `data/` and model files are available locally if needed.

## 📄 License

This project is licensed under the MIT License.

## 🏷️ Tags

machine-learning, regression, steam, game-sales, python, xgboost, scikit-learn, pandas
