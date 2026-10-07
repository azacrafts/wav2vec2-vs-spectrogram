# Wav2Vec2 vs. Spectrogram Models for Audio Classification

A deep learning project comparing traditional spectrogram-based audio classifiers with a speech-pretrained **Wav2Vec2** model.

The goal was to evaluate how different audio representations and model architectures perform across matched and mismatched domains, including robustness under noise.

## Overview

The project compares four approaches to audio classification:

- MFCC features + Random Forest
- Log-mel spectrogram CNN
- ResNet-18 on spectrogram representations
- Wav2Vec2 on raw audio waveforms

Experiments were conducted on:

- **ESC-10** — environmental sound classification
- **Speech Commands** — speech classification

This setup allows comparison between a speech-pretrained model and spectrogram-based approaches both inside and outside the model's original domain.

## Models

### MFCC + Random Forest

A classical machine learning baseline using Mel-Frequency Cepstral Coefficients as audio features.

### Log-Mel CNN

A convolutional neural network trained on log-mel spectrogram representations.

### ResNet-18

An ImageNet-pretrained ResNet-18 adapted to classify audio represented as spectrogram images.

### Wav2Vec2

A pretrained speech representation model operating directly on raw audio waveforms.

## Datasets

### ESC-10

Environmental sound classification dataset containing categories of non-speech audio.

It serves as a mismatched domain for Wav2Vec2, which was originally pretrained on speech.

### Speech Commands

Speech dataset containing short spoken commands.

This represents a domain more closely aligned with Wav2Vec2 pretraining.

## Evaluation

The models were compared using several criteria:

- Classification accuracy
- Performance under added noise
- Model calibration
- Computational cost
- Behavior across matched and mismatched domains

## Experimental Pipeline

```text
Audio Dataset
     ↓
Preprocessing
     ↓
 ┌───────────────────────────────┐
 │                               │
MFCC                         Raw Waveform
 │                               │
Random Forest                 Wav2Vec2
 │                               │
 └──────────────┬────────────────┘
                │
        Spectrogram Models
        ├── Log-Mel CNN
        └── ResNet-18
                │
                ↓
        Model Evaluation
                │
                ↓
Accuracy / Noise Robustness /
Calibration / Computational Cost
```

## Tech Stack

- Python
- PyTorch
- Wav2Vec2
- ResNet-18
- Convolutional Neural Networks
- Scikit-learn
- MFCC
- Log-Mel Spectrograms
- Jupyter Notebook

## Project Structure

```text
wav2vec2-vs-spectrogram/
├── notebooks/        # Experiments and analysis
├── results/          # Experimental results
├── src/              # Core implementation
├── Makefile          # Project commands
├── requirements.txt  # Python dependencies
└── README.md
```

## Running the Project

Clone the repository:

```bash
git clone https://github.com/azacrafts/wav2vec2-vs-spectrogram.git
cd wav2vec2-vs-spectrogram
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The experiment notebooks and analysis can then be found in the `notebooks/` directory.

## Project Context

This project was developed as part of a **CSCI 494 deep learning course project**.

The main objective was to compare general-purpose spectrogram-based models with a speech-pretrained raw-waveform model and study how their performance changes across domains and under noisy conditions.

## Key Questions

The project investigates:

- Does speech pretraining provide an advantage over spectrogram-based models?
- How well does Wav2Vec2 generalize to environmental sounds?
- How robust are the different architectures to noise?
- How do accuracy and calibration change under distribution shift?
- What trade-offs exist between predictive performance and computational cost?

## Future Improvements

- Evaluate additional self-supervised audio models
- Test on larger audio datasets
- Add more noise conditions and corruption types
- Compare inference latency and memory usage
- Perform more detailed calibration analysis
- Add cross-dataset transfer learning experiments
