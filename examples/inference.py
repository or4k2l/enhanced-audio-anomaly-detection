from detector_factory import DetectorFactory

# Example usage of the factory pattern
class AnomalyDetector:
    def __init__(self, model_type):
        self.model = DetectorFactory.create_detector(model_type)

# Inference logic continues...