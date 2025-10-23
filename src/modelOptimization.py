import tensorflow as tf
import numpy as np
import os


class ModelOptimizer:
    def __init__(self, model):
        self.model = model
        self.tfliteModelFp32 = None
        self.tfliteModelQuantInt8 = None
        self.tfliteModelDynamic = None
        
    def convertToTfLiteFp32(self, outputPath='thinker_model_fp32.tflite'):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        self.tfliteModelFp32 = converter.convert()
        
        with open(outputPath, 'wb') as f:
            f.write(self.tfliteModelFp32)
        
        print(f"Float32 TFLite model saved as '{outputPath}'")
        return outputPath
    
    def representativeDatasetGen(self, xTrain):
        def generator():
            for i in range(min(500, len(xTrain))):
                yield [xTrain[i:i+1].astype(np.float32)]
        return generator
    
    def convertToTfLiteDynamicQuant(self, outputPath='thinker_model_dynamic_quant.tflite'):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        self.tfliteModelDynamic = converter.convert()
        
        with open(outputPath, 'wb') as f:
            f.write(self.tfliteModelDynamic)
        
        print(f"Dynamic Quantized TFLite model saved as '{outputPath}'")
        return outputPath
    
    def convertToTfLiteQuantInt8(self, xTrain, outputPath='thinker_model_quant_int8.tflite'):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representativeDatasetGen(xTrain)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        self.tfliteModelQuantInt8 = converter.convert()
        
        with open(outputPath, 'wb') as f:
            f.write(self.tfliteModelQuantInt8)
        
        print(f"Quantized INT8 TFLite model saved as '{outputPath}'")
        return outputPath
    
    def validateQuantizedModel(self, xTest, yTest, modelPath='thinker_model_quant_int8.tflite'):
        interpreter = tf.lite.Interpreter(model_path=modelPath)
        interpreter.allocate_tensors()
        
        inputDetails = interpreter.get_input_details()
        outputDetails = interpreter.get_output_details()
        
        yPredQuant = []
        yPredProba = []
        
        for xSample in xTest:
            if inputDetails[0]['dtype'] == np.int8:
                inputScale, inputZeroPoint = inputDetails[0]['quantization']
                xSampleQuantized = (xSample / inputScale + inputZeroPoint).astype(np.int8)
                interpreter.set_tensor(inputDetails[0]['index'], 
                                      np.expand_dims(xSampleQuantized, axis=0))
            else:
                interpreter.set_tensor(inputDetails[0]['index'], 
                                      np.expand_dims(xSample.astype(np.float32), axis=0))
            
            interpreter.invoke()
            
            output = interpreter.get_tensor(outputDetails[0]['index'])
            
            if outputDetails[0]['dtype'] == np.int8:
                outputScale, outputZeroPoint = outputDetails[0]['quantization']
                output = (output.astype(np.float32) - outputZeroPoint) * outputScale
            
            probability = float(output[0])
            yPredProba.append(probability)
            prediction = 1 if probability > 0.5 else 0
            yPredQuant.append(prediction)
        
        return yPredQuant, yPredProba
    
    def findOptimalThreshold(self, yTest, yPredProba):
        from sklearn.metrics import f1_score
        
        thresholds = np.arange(0.3, 0.8, 0.05)
        bestThreshold = 0.5
        bestF1 = 0
        
        for threshold in thresholds:
            yPredTemp = (np.array(yPredProba) > threshold).astype(int)
            f1 = f1_score(yTest, yPredTemp, pos_label=1)
            
            if f1 > bestF1:
                bestF1 = f1
                bestThreshold = threshold
        
        print(f"\nOptimal threshold: {bestThreshold:.2f} (F1-Score: {bestF1:.4f})")
        return bestThreshold
    
    def compareFileSizes(self, fp32Path='thinker_model_fp32.tflite', 
                        quantPath='thinker_model_quant_int8.tflite',
                        dynamicPath='thinker_model_dynamic_quant.tflite'):
        fp32Size = os.path.getsize(fp32Path) / 1024 if os.path.exists(fp32Path) else 0
        quantSize = os.path.getsize(quantPath) / 1024 if os.path.exists(quantPath) else 0
        dynamicSize = os.path.getsize(dynamicPath) / 1024 if os.path.exists(dynamicPath) else 0
        
        print(f"\nFile Size Comparison:")
        print(f"Float32 TFLite model size: {fp32Size:.2f} KB")
        
        if dynamicSize > 0:
            print(f"Dynamic Quantized TFLite model size: {dynamicSize:.2f} KB")
            print(f"Dynamic size reduction: {(1 - dynamicSize / fp32Size) * 100:.2f}%")
        
        if quantSize > 0:
            print(f"INT8 Quantized TFLite model size: {quantSize:.2f} KB")
            sizeReduction = (1 - quantSize / fp32Size) * 100
            print(f"INT8 size reduction: {sizeReduction:.2f}%")
        
        return {
            'fp32Size': fp32Size,
            'quantSize': quantSize,
            'dynamicSize': dynamicSize,
            'sizeReduction': (1 - quantSize / fp32Size) * 100 if fp32Size > 0 else 0
        }
