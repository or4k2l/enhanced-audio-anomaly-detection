class AnomalyDetector:
    """Abstract base class for anomaly detectors."""

    def train(self):
        """Trains the detector.""" 
        # training logic here

    def predict(self):
        """Makes predictions.""" 
        # prediction logic here

    def is_fitted(self):
        """Returns whether the model is fitted.""" 
        # logic to check if model is fitted

# additional code here