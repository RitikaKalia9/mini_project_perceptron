# Binary Pattern Classifier using Perceptron

A simple Streamlit application that demonstrates how a single-layer perceptron learns and predicts binary logic gates (AND, OR, NAND, NOR and NOT) using PyTorch.

## Features

- Train a perceptron for AND, OR, NAND, NOR and NOT gates
- Save trained model weights and bias
- Load trained models without retraining
- Predict output for binary inputs
- Display sigmoid probability
- Show learned weights and bias
- Linearly-separable plot for each gate: the four truth-table points plus the decision line the perceptron learned, with your selected input ringed
- Explains why XOR is not supported (a single-layer perceptron can't learn XOR: it isn't linearly separable)
- Visualize the forward pass using an animated neural network diagram

## Project Structure

```
.
├── models/
│   ├── perceptron_AND.pt
│   ├── perceptron_OR.pt
│   ├── perceptron_NAND.pt
│   ├── perceptron_NOR.pt
│   ├── perceptron_NOT.pt
│   └── perceptron_<GATE>_meta.json  (one per gate)
├── perceptron_train.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- PyTorch
- Streamlit
- Matplotlib / NumPy

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Note: NOT is a one-input gate. In the app it uses Input 1 only and ignores Input 2, so its decision line is vertical.

## Train the Models

```bash
python perceptron_train.py
```

This trains the perceptron for all four gates and saves the trained models in the `models` folder.

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

