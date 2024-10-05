

# Match Winner Prediction and Model Evaluation

This project is a web application built using **Streamlit** to predict the winner of an ODI cricket match based on historical data. It leverages machine learning models like Decision Trees and Random Forests for classification and provides comprehensive model evaluation metrics.

## Features

- **Match Winner Prediction**: Users can select two teams and a venue to predict the match winner.
- **Data Display**: Option to view the dataset used for training the model.
- **Model Selection**: Choose between Decision Tree or Random Forest classifiers.
- **Hyperparameter Tuning**: Utilizes GridSearchCV for tuning the model's hyperparameters.
- **Model Evaluation**: Provides key metrics like accuracy, confusion matrix, and classification report.
- **Overfitting/Underfitting Detection**: Displays training and testing accuracy to assess model performance.
- **Visualizations**: Heatmaps for confusion matrix and comparison of classification metrics.

## Dataset

The app uses a CSV file (`ODI_Match_info.csv`) containing information about past ODI matches, including:

- Teams (`team1`, `team2`)
- Venue
- Match outcome (winner)

## How It Works

1. **Load Data**: The dataset is loaded and cached for efficient re-use.
2. **Preprocessing**:
   - Teams and venues are encoded using Label Encoding.
   - Features are extracted and split into training and testing sets.
3. **Feature Selection**: Recursive Feature Elimination (RFE) is used to select the most relevant features.
4. **Model Training**: Users can choose a model, which is trained with the best parameters from GridSearchCV.
5. **Prediction**: The app predicts the match winner based on user inputs for team and venue.

## Models Used

- **Decision Tree Classifier**
- **Random Forest Classifier**

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/match-winner-prediction.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Libraries Used

- **Streamlit** for building the web interface.
- **Pandas** for data manipulation.
- **Seaborn** and **Matplotlib** for visualizations.
- **Scikit-learn** for machine learning models and evaluation.

