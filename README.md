# Translate-AI 🌍

A sophisticated bidirectional English-German translation system powered by transformers, featuring real-time speech translation with voice synthesis capabilities.

## Features ✨

- **Bidirectional Translation**: English ↔ German translation
- **Speech Recognition**: Real-time voice input in both languages
- **Text-to-Speech**: Audio output for translations
- **Advanced Preprocessing**: Data cleaning and augmentation pipeline
- **Model Training**: Custom fine-tuning on German-English datasets
- **Cross-Validation**: K-fold validation for robust model evaluation

## Project Structure 📁

```
Translate-AI/
├── data/
│   ├── raw/                 # Raw dataset files
│   └── processed/           # Preprocessed training data
├── saved_models/            # Trained model checkpoints
├── preprocessing.py         # Data preprocessing pipeline
├── dataset.py              # Dataset download utilities
├── train_algorithm.py      # Model training script
├── Translator.py           # Main translation interface
└── README.md               # This file
```

## Installation 🚀

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Microphone for speech input
- Speakers/headphones for audio output

### Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/Translate-AI.git
    cd Translate-AI
    ```

2. **Install dependencies**
    ```bash
    pip install torch transformers pandas tqdm nltk gTTS pygame SpeechRecognition pyaudio sacremoses keyboard scikit-learn wandb sacrebleu
    ```

3. **Download the dataset**
    ```bash
    python dataset.py
    ```

4. **Preprocess the data**
    ```bash
    python preprocessing.py
    ```

## Usage 💡

### Quick Start - Translation Chat

```bash
python Translator.py
```

**Interactive Commands:**
- Press `Enter` to start speaking
- Type `switch` to change translation direction
- Type `quit` to exit

### Training Your Own Model

1. **Prepare your data** (automatically done by preprocessing.py)
2. **Start training**
    ```bash
    python train_algorithm.py
    ```

### Custom Training Configuration

Modify the `TrainingConfig` class in `train_algorithm.py`:

```python
@dataclass
class TrainingConfig:
    model_name: str = "Helsinki-NLP/opus-mt-en-de"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    lr: float = 5e-5
    # ... other parameters
```

## Dataset 📊

The project uses the **Tatoeba Corpus** - a multilingual collection of sentences and translations:

- **Source**: https://tatoeba.org/
- **Languages**: German ↔ English
- **Size**: ~1M+ sentence pairs
- **Format**: TSV (Tab-separated values)

### Dataset Statistics
- Average English sentence length: ~8.5 words
- Average German sentence length: ~8.2 words
- Vocabulary size: ~50k words per language

## Model Architecture 🏗️

### Base Model
- **Architecture**: MarianMT (Transformer-based)
- **Pre-trained**: Helsinki-NLP/opus-mt-en-de
- **Fine-tuning**: Custom German-English corpus

### Training Features
- **Mixed Precision**: FP16 for faster training
- **Gradient Accumulation**: Effective batch size scaling
- **Cross-Validation**: 5-fold validation
- **Early Stopping**: Prevents overfitting
- **BLEU Scoring**: Translation quality metrics

## Performance Metrics 📈

| Metric | Score |
|--------|-------|
| BLEU Score | 28.5+ |
| Training Loss | <0.5 |
| Validation Loss | <0.6 |
| Translation Speed | ~50ms per sentence |

## Configuration ⚙️

### Preprocessing Settings

```python
preprocessor = AdvancedPreprocessor(
    input_file="./data/ge-en.tsv",
    output_dir="data/processed/",
    min_length=3,
    max_length=100,
    test_size=0.1,
    val_size=0.1,
    random_state=42
)
```

### Translation Settings

- **Max Length**: 128 tokens
- **Beam Search**: 4 beams
- **Length Penalty**: 0.6
- **Early Stopping**: Enabled

## API Reference 🔧

### TranslationChat Class

```python
chat = TranslationChat(model_path="saved_models/best_model_epoch_1")

# Translate text
translation = chat.translate("Hello world", direction="en2de")

# Speech-to-speech translation
chat.chat()  # Interactive mode
```

### AdvancedPreprocessor Class

```python
preprocessor = AdvancedPreprocessor(input_file="data.tsv")
preprocessor.process()  # Complete pipeline
```

## Troubleshooting 🛠️

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch_size in TrainingConfig
   - Enable gradient accumulation

2. **Audio Issues**
   - Install pyaudio: `pip install pyaudio`
   - Check microphone permissions

3. **Model Loading Errors**
   - Ensure model path exists
   - Check torch/transformers versions

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: 4GB VRAM minimum for training
- **Storage**: 5GB for models and data

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments 🙏

- **Tatoeba Project** for the translation corpus
- **Hugging Face** for transformer models
- **Helsinki-NLP** for pre-trained MarianMT models
- **OpenAI** for inspiration and guidance


---

**Made with ❤️ for the open-source community**
