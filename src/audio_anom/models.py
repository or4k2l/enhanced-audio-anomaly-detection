import joblib
import numpy as np

class AnomalyDetector:
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

class RandomForestAnomalyDetector(AnomalyDetector):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.model = None
        self.is_fitted = False

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(random_state=self.random_state)
        self.model.fit(X, y)
        self.is_fitted = True
        self.best_estimator_ = self.model

    def save(self, file_path):
        joblib.dump({'best_estimator_': self.best_estimator_, 'random_state': self.random_state, 'is_fitted': self.is_fitted}, file_path)

    def load(self, file_path):
        data = joblib.load(file_path)
        self.best_estimator_ = data['best_estimator_']
        self.random_state = data['random_state']
        self.is_fitted = data['is_fitted']

    def predict(self, X):
        if not self.is_fitted:
            raise Exception('Model is not fitted yet. Call fit() before predicting.')
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise Exception('Model is not fitted yet. Call fit() before predicting.')
        return self.best_estimator_.predict_proba(X)

class XGBoostAnomalyDetector(AnomalyDetector):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.model = None
        self.is_fitted = False

    def fit(self, X, y):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X, y)
        self.is_fitted = True
        self.best_estimator_ = self.model

    def save(self, file_path):
        joblib.dump({'best_estimator_': self.best_estimator_, 'random_state': self.random_state, 'is_fitted': self.is_fitted}, file_path)

    def load(self, file_path):
        data = joblib.load(file_path)
        self.best_estimator_ = data['best_estimator_']
        self.random_state = data['random_state']
        self.is_fitted = data['is_fitted']

    def predict(self, X):
        if not self.is_fitted:
            raise Exception('Model is not fitted yet. Call fit() before predicting.')
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise Exception('Model is not fitted yet. Call fit() before predicting.')
        return self.best_estimator_.predict_proba(X)

class AutoencoderAnomalyDetector(AnomalyDetector):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.model = None
        self.is_fitted = False

    def fit(self, X):
        from keras.models import Sequential
        from keras.layers import Dense
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=(X.shape[1],)))
        self.model.add(Dense(X.shape[1], activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, X, epochs=50, batch_size=32, verbose=0)
        self.is_fitted = True

    def save(self, file_path):
        joblib.dump({'best_estimator_': self.model, 'random_state': self.random_state, 'is_fitted': self.is_fitted}, file_path)

    def load(self, file_path):
        data = joblib.load(file_path)
        self.model = data['best_estimator_']
        self.random_state = data['random_state']
        self.is_fitted = data['is_fitted']

    def predict(self, X):
        if not self.is_fitted:
            raise Exception('Model is not fitted yet. Call fit() before predicting.')
        reconstructed = self.model.predict(X, verbose=0)
        return np.mean(np.power(X - reconstructed, 2), axis=1)