import json
import numpy as np
from src.dataPreprocessing import DataPreprocessor


def exportScalerParams():
    preprocessor = DataPreprocessor('features_with_labels.csv')
    preprocessor.loadData()
    preprocessor.splitData()
    
    scalerParams = {
        'means': preprocessor.scaler.mean_.tolist(),
        'stds': preprocessor.scaler.scale_.tolist(),
        'feature_count': len(preprocessor.scaler.mean_),
        'description': 'StandardScaler parameters for Guardian AI fall detection model',
        'usage': 'normalized_feature[i] = (raw_feature[i] - means[i]) / stds[i]'
    }
    
    with open('scaler_params.json', 'w') as f:
        json.dump(scalerParams, f, indent=2)
    
    print(f"Exported scaler parameters to scaler_params.json")
    print(f"Features: {scalerParams['feature_count']}")
    print(f"Mean range: [{min(scalerParams['means']):.4f}, {max(scalerParams['means']):.4f}]")
    print(f"Std range: [{min(scalerParams['stds']):.4f}, {max(scalerParams['stds']):.4f}]")
    
    return scalerParams


if __name__ == '__main__':
    exportScalerParams()
