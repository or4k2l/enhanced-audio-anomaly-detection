import numpy as np
import pandas as pd
from audio_anom import feature_extraction, model_training, evaluation
import matplotlib.pyplot as plt

# Load the dataset
try:
    print('📥 Loading dataset...')
    dataset = pd.read_csv('path/to/your/dataset.csv')  # Update with your dataset path
    print('✅ Dataset loaded successfully')
except Exception as e:
    print(f'❌ Error loading dataset: {str(e)}')
    exit()

# Split the dataset
try:
    print('🔍 Splitting dataset...')
    train_data, test_data = model_training.split_dataset(dataset)
    print('✅ Dataset split into training and testing sets')
except Exception as e:
    print(f'❌ Error splitting dataset: {str(e)}')
    exit()

# Feature extraction
try:
    print('🔧 Extracting features...')
    X_train, y_train = feature_extraction.extract_features(train_data)
    X_test, y_test = feature_extraction.extract_features(test_data)
    print('✅ Features extracted successfully')
except Exception as e:
    print(f'❌ Error during feature extraction: {str(e)}')
    exit()

# Model training with Random Forest
try:
    print('🛠️ Training Random Forest model...')
    rf_model = model_training.train_random_forest(X_train, y_train)
    print('✅ Random Forest model trained successfully')
except Exception as e:
    print(f'❌ Error during Random Forest training: {str(e)}')
    exit()

# Model training with XGBoost
try:
    print('🛠️ Training XGBoost model...')
    xgb_model = model_training.train_xgboost(X_train, y_train)
    print('✅ XGBoost model trained successfully')
except Exception as e:
    print(f'❌ Error during XGBoost training: {str(e)}')
    exit()

# Evaluation
try:
    print('📊 Evaluating models...')
    rf_results = evaluation.evaluate_model(rf_model, X_test, y_test)
    xgb_results = evaluation.evaluate_model(xgb_model, X_test, y_test)
    print('✅ Model evaluation completed')
except Exception as e:
    print(f'❌ Error during model evaluation: {str(e)}')
    exit()

# Visualization
try:
    print('📈 Visualizing results...')
    plt.figure(figsize=(10, 5))
    plt.plot(rf_results['metric'], label='Random Forest')
    plt.plot(xgb_results['metric'], label='XGBoost')
    plt.title('Model Evaluation Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    print('✅ Visualization completed')
except Exception as e:
    print(f'❌ Error during visualization: {str(e)}')
    exit()  

print('🎉 Script completed successfully!')