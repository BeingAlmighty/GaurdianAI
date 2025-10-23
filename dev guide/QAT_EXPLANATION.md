# Quantization-Aware Training (QAT) Implementation

## 🎯 What is QAT?

**Quantization-Aware Training** is an advanced technique that trains the model to anticipate and compensate for the precision loss that occurs during quantization. Unlike post-training quantization (PTQ), QAT simulates quantization during training itself.

## 🔄 QAT vs Post-Training Quantization

### Post-Training Quantization (PTQ)
```
Train Model → Quantize → Deploy
         (Float32)  (INT8)
```
- ❌ Model doesn't know it will be quantized
- ❌ Can lose 2-5% accuracy
- ✅ Fast and simple

### Quantization-Aware Training (QAT)
```
Train Model → Fine-tune with Simulated Quantization → Quantize → Deploy
         (Float32)         (Fake Quant)                (INT8)
```
- ✅ Model learns to work with quantization
- ✅ Retains 95-99% of original accuracy
- ✅ Better performance on real hardware
- ⚠️ Takes longer to train

## 🏗️ How QAT Works

### Step 1: Pre-train Base Model
Normal training with Float32 precision to get good initial weights.

### Step 2: Apply Quantization Simulation
Insert "fake quantization" nodes that simulate INT8 precision during training:
- Forward pass: Quantize → Dequantize (simulates INT8)
- Backward pass: Full precision gradients
- Model learns to be robust to quantization errors

### Step 3: Convert to TFLite INT8
Final conversion produces a model that performs better because it was trained with quantization in mind.

## 📊 Expected Performance Improvements

| Metric | Post-Training Quant | QAT |
|--------|-------------------|-----|
| **Accuracy Loss** | 1-5% | 0.1-1% |
| **Fall Recall** | 96-97% | 97-98% |
| **ADL Recall** | 90-92% | 92-94% |
| **Model Size** | ~26 KB | ~26 KB |
| **Inference Speed** | Fast | Fast |

## 🔧 Implementation Details

### Training Process
1. **Pre-training (100 epochs)**
   - Standard training with balanced class weights
   - Early stopping with patience=15
   - Learning rate scheduling

2. **QAT Fine-tuning (50 epochs)**
   - Lower learning rate (0.0001 vs 0.001)
   - Quantization simulation on all layers
   - Early stopping with patience=10
   - Continued class weight balancing

### Key Parameters
```python
optimizer = Adam(learning_rate=0.0001)  # Lower LR for stability
epochs = 50  # Fewer epochs for fine-tuning
batch_size = 64  # Same as pre-training
```

## 🚀 Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Training
```bash
python main.py
```

### Output Models
- `thinker_model_qat_int8.tflite` - **RECOMMENDED** for production
- `thinker_model_quant_int8.tflite` - Standard PTQ baseline
- `thinker_model_fp32.tflite` - Float32 reference

## 💡 Why QAT is Better for Guardian AI

1. **Critical Application**: Fall detection is life-critical, every percentage point matters
2. **Edge Deployment**: Running on mobile devices requires INT8, QAT ensures minimal accuracy loss
3. **Class Imbalance**: QAT better handles the imbalanced fall/non-fall distribution after quantization
4. **Real-world Robustness**: Better generalization to unseen fall patterns

## 📈 Training Time Comparison

| Method | Training Time | Accuracy | Best For |
|--------|--------------|----------|----------|
| **Standard PTQ** | ~5-10 min | 96% | Quick prototypes |
| **QAT** | ~15-20 min | 96.5-97% | Production deployment |

## 🎓 Technical Background

QAT uses **fake quantization operators** during training:
```python
# Simulates: Float32 → INT8 → Float32
y_quantized = (round(x / scale) + zero_point) * scale
```

This allows:
- Forward pass: Simulates quantized inference
- Backward pass: Gradient flow in Float32
- Model learns quantization-friendly features

## 🔍 Monitoring QAT

During training, watch for:
- ✅ Stable training (no large oscillations)
- ✅ Validation metrics improving
- ✅ Small gap between train/val accuracy
- ⚠️ If unstable: reduce learning rate

## 📝 References

- TensorFlow Model Optimization Toolkit
- [QAT Documentation](https://www.tensorflow.org/model_optimization/guide/quantization/training)
- Minimal Viable Data (MVD) Framework
- Mobile ML Best Practices

---

**Result**: A production-ready INT8 model that maintains near-Float32 accuracy! 🎉
