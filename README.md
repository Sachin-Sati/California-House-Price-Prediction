# California House Price Prediction

A complete end-to-end Machine Learning project that predicts median house prices in California.

![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-success?style=flat&logo=streamlit)

## Project Overview

This project builds a regression model to predict the median house value based on various features such as median income, house age, rooms, location and more. 

It emphasizes clean code, feature, enginerring, feature scaling, model evaluation, interpretability and deployment.

### Key Highlights:

- Exploratory Data Analysis (EDA)
- Data Preprocessing (Feature Engineering, Feature Scaling)
- Comparison of multiple Machine Learning models
- Hyperparameter Tuning (*to be added)
- Deployment using Streamlit

## Tech Stack:

- **Languages**: Python 3.14.4
- **Data Manipulation**: pandas, numpy
- **Data Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost
- **Deployment**: streamlit 

## Project Structure

```
house_price_prediction/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── utils.py
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
├── app.py                      # streamlit web app
├── main.py                     # run full ml pipeline
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/Sachin-Sati/California-House-Price-Prediction
```

## Results & Insights

- Median Income is the strongest predictor of house price.
- Engineered features significantly improved model performance.
- SHAP analysis shows clear feature impact and interactions.

## Future Improvements

- Experiment with advanced ensemble techniques (Stacking)
- Add more sophisticated feature engineering
- Implement model monitoring and logging
- Deploy on cloud (AWS/Heroku/Render)
- Add Docker support

##  License

This project is licensed under the MIT License - see the LICENSE file for details.