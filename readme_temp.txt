---
license: cc-by-4.0
task_categories:
- image-classification
language:
- en
pretty_name: ASL Static Landmarks v1
tags:
- asl
- hand-landmarks
- mediapipe
- sign-language
---
# ASL Static Letter Recognizer (Real-Time)

Real-time ASL fingerspelling recognizer built with **MediaPipe + TensorFlow**.

This project recognizes **24 static ASL letters** from webcam input.
- ✅ Included: static letters (e.g., A, B, C, ...)
- ❌ Excluded: **J, Z** (motion-based letters)

## Current model status

- Final static model: `models/asl_static_model.h5`
- Label encoder: `models/static_label_encoder.pkl`
- Input features: **86 engineered features** from 21 hand landmarks
- Reported accuracy: **92.74% overall**, **94.28% per-letter average**

## Project structure

```
aisl/
├── src/
│   ├── hand_detector.py
│   ├── feature_engineer.py
│   ├── data_collector.py
│   ├── data_processor.py
│   ├── train_static_only.py
│   ├── analyze_model.py
│   └── predictor_static.py
├── models/
│   ├── asl_static_model.h5
│   └── static_label_encoder.pkl
├── data/
│   ├── processed/
│   ├── processed_v2/
│   └── raw/ (ignored by git)
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

	`pip install -r requirements.txt`

## Run live prediction

From project root:

`python3 src/predictor_static.py`

### Controls

- `q`: quit
- `+` / `-`: increase/decrease confidence threshold

## Training workflow (optional)

1. Collect data (raw landmarks)
2. Engineer features and prepare dataset
3. Train static model
4. Analyze confusion and per-letter performance

Primary scripts:
- `src/collect_more_static.py`
- `src/feature_engineer.py`
- `src/train_static_only.py`
- `src/analyze_model.py`

## Notes

- The current recognizer is optimized for **single-hand static poses**.
- For full alphabet support, add a temporal model (e.g., LSTM/GRU/Transformer) for motion letters **J** and **Z**.
- Performance depends on lighting, camera quality, and consistent hand positioning.

