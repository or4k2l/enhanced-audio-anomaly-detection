import joblib

class ModelExporter:
    @staticmethod
    def export_model_package(model, scaler, pca, feature_cols, config, performance_metrics, output_path):
        package_dict = {
            'model': model,
            'scaler': scaler,
            'pca': pca,
            'feature_columns': feature_cols,
            'config': config,
            'performance_metrics': performance_metrics
        }
        joblib.dump(package_dict, str(output_path))
