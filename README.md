# 🧠 BERT Next Sentence Predictor

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.40.2-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-FF4B4B.svg)](https://streamlit.io/)

A deep learning project that uses BERT (Bidirectional Encoder Representations from Transformers) to predict whether two sentences logically follow each other. This implementation includes model training, evaluation, and an interactive Streamlit web application for real-time predictions.

![BERT NSP Demo](https://img.shields.io/badge/Status-Active-success)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/PriscillajospinG/next-sentence-predictor-BERT.git
cd next-sentence-predictor-BERT

# 2. Install dependencies
pip install transformers==4.40.2 torch pandas scikit-learn matplotlib streamlit

# 3. Download the pre-trained model from Google Drive
# Link: https://drive.google.com/drive/folders/1JbDl2axfxqIloN7CI-K8M6wO9Suxyhqe?usp=sharing

# 4. Run predictions
python -c "
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

model = BertForNextSentencePrediction.from_pretrained('./bert_nsp_model')
tokenizer = BertTokenizer.from_pretrained('./bert_nsp_model')

inputs = tokenizer('Hello!', 'Nice to meet you.', return_tensors='pt')
outputs = model(**inputs)
print('Prediction:', 'IsNext ✅' if torch.argmax(outputs.logits) == 0 else 'NotNext ❌')
"
```

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pre-trained Model](#pre-trained-model)
- [Web Application](#web-application)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [License](#license)

## 🎯 Overview

Next Sentence Prediction (NSP) is a binary classification task where the model determines if sentence B logically follows sentence A. This project fine-tunes BERT on a custom dataset and provides:
- PyTorch-based training pipeline
- Comprehensive model evaluation with metrics and visualizations
- Interactive Streamlit web interface for testing predictions
- Model persistence and loading capabilities

## ✨ Features

- **Custom BERT Fine-tuning**: Manual training loop implementation using PyTorch
- **Evaluation Metrics**: 
  - Accuracy and loss tracking
  - Confusion matrix visualization
  - Confidence scores for predictions
- **Interactive Web App**: Streamlit-based UI for real-time sentence pair testing
- **Model Persistence**: Save and load trained models
- **GPU Support**: Automatic GPU detection and utilization
- **Visualization**: Training metrics and confusion matrix plots

## 📊 Dataset

The project uses a custom NSP dataset (`nsp_dataset.csv`) containing sentence pairs with binary labels.

### Dataset Structure:
| Column | Description | Type |
|--------|-------------|------|
| `sentence_a` | First sentence | String |
| `sentence_b` | Second sentence | String |
| `label` | Relationship label | Integer (0/1) |

### Labels:
- **0 (IsNext)**: Sentence B logically follows Sentence A
- **1 (NotNext)**: Sentence B does not follow Sentence A

### Example Data Pairs:
```
Sentence A: "She studied all night for her exams."
Sentence B: "She passed with flying colors." 
→ Label: 0 (IsNext) ✅

Sentence A: "It rained heavily throughout the night."
Sentence B: "He ordered a burger with fries." 
→ Label: 1 (NotNext) ❌
```

## 🏗️ Model Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | `bert-base-uncased` from HuggingFace |
| **Task Head** | Binary classification layer for NSP |
| **Tokenizer** | BERT WordPiece tokenizer |
| **Max Sequence Length** | 64 tokens |
| **Optimizer** | AdamW (learning rate: 2e-5) |
| **Training Epochs** | 3 |
| **Batch Size** | 16 |
| **Device** | CUDA GPU (if available) or CPU |

### Architecture Flow:
```
Input Sentences → BERT Tokenizer → Input IDs + Attention Mask 
→ BERT Encoder → Pooled Output → Classification Head → Softmax 
→ Binary Prediction (IsNext/NotNext)
```

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (optional, but recommended)
- Google Colab account (for notebook execution)

### Dependencies

```bash
pip install transformers==4.40.2
pip install tokenizers==0.19.1
pip install torch
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install streamlit
pip install pyngrok
```

## 🚀 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/PriscillajospinG/next-sentence-predictor-BERT.git
cd next-sentence-predictor-BERT
```

### 2. Run the Jupyter Notebook
Open `NSP.ipynb` in Google Colab or Jupyter Notebook and execute cells sequentially.

### 3. Load Pre-trained Model
```python
import torch
from transformers import BertTokenizer, BertForNextSentencePrediction

model_path = "./bert_nsp_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForNextSentencePrediction.from_pretrained(model_path)
model.eval()
```

### 4. Make Predictions
```python
def predict_nsp(sent_a, sent_b):
    encoding = tokenizer(sent_a, sent_b, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
    
    result = "Is Next Sentence ✅" if pred_label == 0 else "Not Next Sentence ❌"
    return result, probs

# Example
predict_nsp("The sun rises in the east.", "It gives light and warmth to the earth.")
```

## 📈 Training

The training process includes:

1. **Data Loading**: CSV dataset with sentence pairs and labels
2. **Tokenization**: BERT tokenizer with max length 64
3. **Custom Dataset Class**: PyTorch Dataset for batching
4. **Training Loop**: 
   - 3 epochs
   - Batch size: 16
   - Learning rate: 2e-5
   - Loss function: CrossEntropyLoss
5. **Model Checkpointing**: Save best model state

Training code snippet:
```python
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device)
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
```

## 📊 Evaluation

Evaluation metrics include:

- **Loss**: Cross-entropy loss on validation set
- **Accuracy**: Percentage of correct predictions
- **Confusion Matrix**: Visual representation of true vs predicted labels

Example evaluation output:
```
✅ Evaluation Complete | Loss: 0.XXXX | Accuracy: 0.XXXX
```

Confusion Matrix visualization shows:
- True Positives (IsNext correctly predicted)
- True Negatives (NotNext correctly predicted)
- False Positives
- False Negatives

## 💾 Pre-trained Model

🎯 **Ready-to-use trained model available for download!**

### 📁 Download Link:
**[📥 Download BERT NSP Model from Google Drive](https://drive.google.com/drive/folders/1JbDl2axfxqIloN7CI-K8M6wO9Suxyhqe?usp=sharing)**

### 📦 Model Files Included:
```
bert_nsp_model/
├── config.json              # Model configuration & hyperparameters
├── pytorch_model.bin        # Trained model weights (~420MB)
├── tokenizer_config.json    # Tokenizer settings
├── vocab.txt                # BERT vocabulary (30,522 tokens)
└── special_tokens_map.json  # Special tokens ([CLS], [SEP], etc.)
```

### 🔧 How to Load the Model:

**Step 1: Download the model folder from Google Drive**

**Step 2: Load in your Python code:**
```python
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

# Specify path to downloaded model
model_path = "/path/to/bert_nsp_model"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForNextSentencePrediction.from_pretrained(model_path)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("✅ Model loaded successfully!")
```

**Step 3: Make predictions:**
```python
def predict(sentence_a, sentence_b):
    inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
    
    return "IsNext ✅" if prediction == 0 else "NotNext ❌", probs

# Test it
result, confidence = predict(
    "The weather was perfect.",
    "We decided to go for a walk."
)
print(f"Prediction: {result}")
print(f"Confidence: {confidence}")
```

## 🌐 Web Application

An interactive Streamlit web application is included for easy testing:

### Features:
- Text input for two sentences
- Real-time prediction
- Confidence scores display
- User-friendly interface

### Running the App:

1. **Local Deployment**:
```bash
streamlit run app.py
```

2. **Google Colab with ngrok**:
```python
from pyngrok import ngrok
import streamlit as st

# Start Streamlit in background
!nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

# Create public URL
public_url = ngrok.connect(addr="8501")
print("🚀 Streamlit app URL:", public_url)
```

### App Interface:
```
🧠 BERT Next Sentence Prediction
Enter two sentences to see if the second logically follows the first.

Sentence A: [Text input area]
Sentence B: [Text input area]

[🔍 Predict Button]

Result: ✅ Is Next Sentence / ❌ Not Next Sentence
Confidence: IsNext=0.XX, NotNext=0.XX
```

## 📈 Results

### Model Performance:
- ✅ **High Training Accuracy**: Effective learning on training set
- ✅ **Strong Validation Performance**: Robust predictions on unseen data
- ⚡ **Fast Inference**: Real-time predictions using optimized BERT
- 📊 **Reliable Confidence Scores**: Probability distributions for decisions

### Sample Predictions:

| Sentence A | Sentence B | Prediction | Confidence |
|------------|------------|------------|------------|
| "She finished her dinner." | "Then she started watching a movie." | ✅ **IsNext** | ~0.95 |
| "It was raining outside." | "I love programming in Python." | ❌ **NotNext** | ~0.89 |
| "The sun rises in the east." | "It gives light and warmth to the earth." | ✅ **IsNext** | ~0.92 |
| "She studied all night." | "The teacher explained the lesson." | ❌ **NotNext** | ~0.87 |

### Visualization Examples:
- 📊 Training loss curves
- 📈 Accuracy progression over epochs
- 🎯 Confusion matrix heatmap
- 📉 Performance metrics dashboard

## 📁 Project Structure

```
next-sentence-predictor-BERT/
│
├── NSP.ipynb                    # Main Jupyter notebook
├── nsp_dataset.csv              # Training dataset (original)
├── nsp_dataset 9.09.25 PM.csv   # Dataset version
├── README.md                    # Project documentation
├── app.py                       # Streamlit web application (generated)
│
└── bert_nsp_model/              # Saved model directory
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── vocab.txt
    └── special_tokens_map.json
```

## 🛠️ Technologies Used

- **Deep Learning Framework**: PyTorch
- **Transformer Library**: HuggingFace Transformers 4.40.2
- **Model**: BERT (bert-base-uncased)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, scikit-learn
- **Web Framework**: Streamlit
- **Tunneling**: ngrok (for Google Colab)
- **Development**: Google Colab, Jupyter Notebook

## 🔮 Future Improvements

### 🎯 Model Enhancements
- [ ] Expand dataset with more diverse sentence pairs (10K+ samples)
- [ ] Implement k-fold cross-validation for robust evaluation
- [ ] Experiment with other transformer models:
  - RoBERTa (more robust training)
  - ALBERT (lighter & faster)
  - DistilBERT (smaller footprint)
- [ ] Fine-tune on domain-specific data (medical, legal, technical texts)

### 🌍 Multilingual Support
- [ ] Add support for multiple languages (Spanish, French, German, etc.)
- [ ] Use multilingual BERT (mBERT) or XLM-RoBERTa
- [ ] Create language-specific datasets

### 🚀 Deployment & Production
- [ ] Deploy to cloud platforms:
  - AWS SageMaker
  - Google Cloud AI Platform
  - Microsoft Azure ML
- [ ] Create REST API with FastAPI
- [ ] Containerize with Docker
- [ ] Set up CI/CD pipeline

### 📊 Performance Optimization
- [ ] Implement ONNX Runtime for faster inference
- [ ] Add model quantization for reduced size
- [ ] Batch prediction optimization
- [ ] Implement caching for repeated queries

### 🛠️ Features & Functionality
- [ ] Add attention weight visualization
- [ ] Implement SHAP/LIME for model explainability
- [ ] Create comprehensive test suite (pytest)
- [ ] Add data augmentation techniques
- [ ] Build Chrome extension for real-time text analysis
- [ ] Integrate with popular writing tools (Grammarly-style)

### 📚 Documentation & Community
- [ ] Add API documentation (Swagger/OpenAPI)
- [ ] Create video tutorials
- [ ] Write blog posts about implementation
- [ ] Add Jupyter notebook tutorials
- [ ] Build community Discord/Slack channel

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Priscilla Jospin G**
- GitHub: [@PriscillajospinG](https://github.com/PriscillajospinG)

## 🙏 Acknowledgments

- 🤗 **HuggingFace** for the incredible Transformers library
- 🧠 **Google Research** for BERT model and groundbreaking research
- 📚 **PyTorch Team** for the flexible deep learning framework
- 🎨 **Streamlit** for the amazing web app framework
- 💻 **Open Source Community** for various tools and libraries
- 📖 **Research Papers**:
  - [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
  - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 📚 References

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv preprint arXiv:1810.04805*.
2. Vaswani, A., et al. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.

## 📞 Contact & Support

- 📧 **Issues**: Open an issue on [GitHub](https://github.com/PriscillajospinG/next-sentence-predictor-BERT/issues)
- 💬 **Discussions**: Start a discussion in the Discussions tab
- 🐛 **Bug Reports**: Please include detailed steps to reproduce

## ⭐ Show Your Support

If you found this project helpful or interesting, please consider:
- ⭐ Starring the repository
- 🍴 Forking for your own experiments
- 📢 Sharing with others who might benefit
- 💬 Providing feedback or suggestions

---

<div align="center">

**Built with ❤️ using BERT and PyTorch**

</div>