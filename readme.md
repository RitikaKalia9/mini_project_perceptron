# Binary Pattern Classifier using Perceptron

A simple Streamlit application that demonstrates how a single-layer perceptron learns and predicts binary logic gates (AND and OR) using PyTorch.

## Features

- Train a perceptron for AND and OR gates
- Save trained model weights and bias
- Load trained models without retraining
- Predict output for binary inputs
- Display sigmoid probability
- Show learned weights and bias
- Visualize the forward pass using an animated neural network diagram

## Project Structure

```
.
├── models/
│   ├── perceptron_AND.pt
│   ├── perceptron_OR.pt
│   ├── perceptron_AND_meta.json
│   └── perceptron_OR_meta.json
├── perceptron_train.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- PyTorch
- Streamlit

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Train the Models

```bash
python perceptron_train.py
```

This trains the perceptron for both AND and OR gates and saves the trained models in the `models` folder.

## Run the Application

```bash
python -m streamlit run streamlit_app.py
```

## How to Use

1. Select the logic gate (AND or OR).
2. Choose the two binary inputs.
3. Click **Predict**.
4. View:
   - Predicted output
   - Sigmoid probability
   - Weighted sum calculation
   - Decision rule
   - Animated neural network visualization
   - Learned weights and bias

## Output

The application displays:

- Predicted class
- Sigmoid output probability
- Weighted sum (z)
- Decision threshold
- Animated forward-pass visualization
- Model parameters

