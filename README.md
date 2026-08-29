# AI Job Salary Prediction

This project uses machine learning to predict the annual salary of AI and technology professionals based on job-related characteristics. It includes a Jupyter notebook for data analysis, preprocessing, feature engineering, model training, and evaluation, along with a trained XGBoost model and a Streamlit application for salary prediction.

## Project Structure

- `notebooks/eda.ipynb`: Main notebook containing data analysis, visualization, preprocessing, feature engineering, model training, and evaluation.
- `src/app.py`: Streamlit application for interactive salary prediction.
- `models/salary_model.pkl`: Trained XGBoost salary prediction model.
- `models/feature_columns.pkl`: Feature columns used by the trained model.
- `models/skill_columns.pkl`: Skill feature columns used during preprocessing.
- `requirements.txt`: List of required Python packages.

## Features

- Loads and analyzes an AI job-market dataset from Kaggle.
- Performs exploratory data analysis and visualization.
- Preprocesses numerical and categorical features.
- Converts required skills into machine-learning features.
- Compares Linear Regression, Random Forest, and XGBoost.
- Uses GridSearchCV and 5-fold cross-validation for hyperparameter tuning.
- Evaluates models using MAE, MSE, RMSE, and R².
- Uses XGBoost as the final prediction model.
- Saves the trained model and preprocessing information.
- Provides an interactive Streamlit application for salary prediction.

## Getting Started

1. Clone the repository:

   ```bash
   git clone <YOUR-GITHUB-REPOSITORY-URL>
   ```

2. Navigate to the project directory:

   ```bash
   cd ai-job-salary-prediction
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from Kaggle and place it inside the `data/` directory.

5. Run the Streamlit application:

   ```bash
   streamlit run src/app.py
   ```

## Usage

- Open the Streamlit application.
- Enter the job-related characteristics.
- Select the required skills.
- Click **Predict Salary**.
- The trained XGBoost model will generate the predicted annual salary.

The complete analysis and model development process can be found in `notebooks/eda.ipynb`.

## Requirements

See `requirements.txt` for all dependencies.

## Dataset

The dataset used in this project was obtained from Kaggle:

[Global AI Job Market & Salary Trends 2025](https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025)

The dataset is used for exploratory data analysis and model training.

The trained model and preprocessing information are included in this repository, so the raw dataset is not required to run the Streamlit application.