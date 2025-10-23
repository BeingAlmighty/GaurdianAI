import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dataPreprocessing import DataPreprocessor


def evaluateTfliteModel(modelPath, dataPath='features_with_labels.csv', verbose=True):
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"Model file not found: {modelPath}")
    
    if not os.path.exists(dataPath):
        raise FileNotFoundError(f"Data file not found: {dataPath}")
    
    preprocessor = DataPreprocessor(dataPath)
    preprocessor.loadData()
    _, _, xTest, _, _, yTest = preprocessor.splitData()
    
    if verbose:
        print(f'\n{"="*70}')
        print(f'TFLite Model Evaluation')
        print(f'{"="*70}')
        print(f'Model: {os.path.basename(modelPath)}')
        print(f'Size: {os.path.getsize(modelPath)/1024:.2f} KB')
        print(f'Test samples: {len(xTest)}')
    
    interpreter = tf.lite.Interpreter(model_path=modelPath)
    interpreter.allocate_tensors()
    
    inputDetails = interpreter.get_input_details()[0]
    outputDetails = interpreter.get_output_details()[0]
    
    if verbose:
        print(f'\nInput quantization:')
        print(f'  dtype: {inputDetails["dtype"]}')
        print(f'  scale: {inputDetails["quantization"][0]:.6f}')
        print(f'  zero_point: {inputDetails["quantization"][1]}')
        print(f'\nOutput quantization:')
        print(f'  dtype: {outputDetails["dtype"]}')
        print(f'  scale: {outputDetails["quantization"][0]:.6f}')
        print(f'  zero_point: {outputDetails["quantization"][1]}')
    
    probs = []
    for x in xTest:
        if inputDetails['dtype'] == np.int8:
            scale, zero = inputDetails['quantization']
            xq = (x / scale + zero).astype(np.int8)
            interpreter.set_tensor(inputDetails['index'], np.expand_dims(xq, 0))
        else:
            interpreter.set_tensor(inputDetails['index'], np.expand_dims(x.astype(np.float32), 0))
        
        interpreter.invoke()
        
        out = interpreter.get_tensor(outputDetails['index'])
        if outputDetails['dtype'] == np.int8:
            oScale, oZero = outputDetails['quantization']
            out = (out.astype(np.float32) - oZero) * oScale
        
        prob = float(out.flatten()[0])
        probs.append(prob)
    
    probs = np.array(probs)
    
    bestF1 = -1
    bestThreshold = 0.5
    bestPreds = None
    
    if verbose:
        print(f'\nSearching optimal threshold...')
    
    for threshold in np.linspace(0.01, 0.99, 99):
        preds = (probs > threshold).astype(int)
        f1 = f1_score(yTest, preds, pos_label=1)
        if f1 > bestF1:
            bestF1 = f1
            bestThreshold = threshold
            bestPreds = preds
    
    accuracy = accuracy_score(yTest, bestPreds)
    cm = confusion_matrix(yTest, bestPreds)
    
    if verbose:
        print(f'\n{"="*70}')
        print(f'RESULTS (Optimal Threshold = {bestThreshold:.3f})')
        print(f'{"="*70}')
        print(f'\nClassification Report:')
        print(classification_report(yTest, bestPreds, target_names=['ADL', 'Fall']))
        print(f'Confusion Matrix:')
        print(cm)
        print(f'\nKey Metrics:')
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  Fall F1-Score: {bestF1:.4f}')
        print(f'  Fall Recall: {cm[1,1]/(cm[1,0]+cm[1,1]):.4f}')
        print(f'  ADL Recall: {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}')
        print(f'{"="*70}\n')
    
    return {
        'model_path': modelPath,
        'model_size_kb': os.path.getsize(modelPath) / 1024,
        'optimal_threshold': bestThreshold,
        'accuracy': accuracy,
        'fall_f1': bestF1,
        'fall_recall': cm[1,1] / (cm[1,0] + cm[1,1]),
        'adl_recall': cm[0,0] / (cm[0,0] + cm[0,1]),
        'confusion_matrix': cm.tolist(),
        'test_samples': len(xTest)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate TFLite model for Guardian AI')
    parser.add_argument('model', type=str, help='Path to .tflite model file')
    parser.add_argument('--data', type=str, default='features_with_labels.csv',
                       help='Path to CSV data file (default: features_with_labels.csv)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    try:
        results = evaluateTfliteModel(args.model, args.data, verbose=not args.quiet)
        
        if args.quiet:
            print(f"Threshold: {results['optimal_threshold']:.3f}, "
                  f"F1: {results['fall_f1']:.4f}, "
                  f"Accuracy: {results['accuracy']:.4f}")
        
        return 0
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
