"""Trains the AND / OR perceptrons and saves them in models/."""

import os
import json
import torch
# torch.nn contains classes for building neural networks such as: linear, sigmoid, BCELoss
import torch.nn as nn

# Folder where trained models will be saved
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Truth tables for the two gates this project supports (AND / OR).
# Same 4-row datasets used in the lab, just organised in a dictionary so we
# can loop over both gates instead of copy-pasting the training loop twice.
# ---------------------------------------------------------------------------
GATES = {
    "AND": {
        "x": torch.tensor([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]
        ]),
        "y": torch.tensor([
            [0.],
            [0.],
            [0.],
            [1.]
        ]),
    },
    "OR": {
        "x": torch.tensor([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]
        ]),
        "y": torch.tensor([
            [0.],
            [1.],
            [1.],
            [1.]
        ]),
    },
}


def train_and_save(gate_name, x, y, epochs=1000, lr=0.5):
    """
    Trains a single-layer perceptron on one gate's truth table and saves
    the result. This is exactly the training loop from the lab notebook,
    just wrapped in a function so it can be reused for AND and OR.
    """

    # Fixed seed so every run (and every deployment) learns the same weights
    torch.manual_seed(42)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # model building
    # nn.Sequential is a pytorch container that stacks layers one after another
    # and computes the weighted sum. Since this is a single-layer perceptron,
    # there are only two layers: layer 0 = one linear layer, layer 1 = one
    # activation function.
    model = nn.Sequential(
        nn.Linear(2, 1),   # fully connected (dense) layer: 2 binary inputs -> 1 output
        nn.Sigmoid()       # squashes the linear output into a 0-1 probability
    )

    # criterion is simply a variable name that stores the loss function.
    # Binary Cross Entropy Loss measures the difference between the
    # model's predicted_output and the actual_output.
    # Larger loss means larger prediction error.
    criterion = nn.BCELoss()

    # The optimizer is responsible for updating the model's weights and bias
    # to reduce the loss. torch.optim.SGD = Stochastic Gradient Descent.
    # model.parameters() tells SGD: update all trainable parameters
    # (weight 1, weight 2, bias). lr (learning rate) controls how big each update is.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Repeat training for 'epochs' passes over the full dataset.
    # One epoch = one complete pass through the entire training dataset.
    losses = []
    for epoch in range(epochs):
        # Clear gradients from the previous iteration (reset to zero).
        optimizer.zero_grad()

        # Forward pass: input -> linear -> sigmoid -> predicted output
        output = model(x)

        # Compute the Binary Cross-Entropy loss by comparing
        # the predicted output with the target output.
        loss = criterion(output, y)

        # Compute gradients of the loss with respect to each weight and bias
        # using backpropagation. This single line activates the autograd engine.
        loss.backward()

        # SGD internally performs: w(new) = w(old) - (lr * gradient)
        # Using the gradients calculated above, the optimizer updates
        # the weights and bias.
        optimizer.step()

        losses.append(loss.item())

        # Print progress every 100 epochs so we can see the loss decreasing
        if epoch % 100 == 0:
            print(f"[{gate_name}] Epoch {epoch}, Loss = {loss.item():.6f}")

    # loss.item() extracts the numerical value from the loss tensor.
    print(f"[{gate_name}] Final Loss:", loss.item())

    # We are only making predictions now, not training -> torch.no_grad()
    with torch.no_grad():
        predictions = model(x)
        print(f"[{gate_name}] Predictions:\n", predictions)

    # ---------------------------------------------------------------------
    # Display the learned weights and bias 
    # model.named_parameters() returns all trainable parameters of the
    # model along with their names. For our perceptron (Linear(2,1)),
    # it returns 'weight' and 'bias'.
    # ---------------------------------------------------------------------
    print(f"\n[{gate_name}] Learned Parameters")
    for name, param in model.named_parameters():
        print(name)
        print(param.data)

    # ---------------------------------------------------------------------
    # SAVE the trained model so the Streamlit app can load it later
    # without retraining. We save two things:
    #   1) the actual model weights (state_dict) -> used to reload the model
    #   2) a small JSON with the readable numbers -> easy to display in the app
    # ---------------------------------------------------------------------
    weights_path = os.path.join(MODELS_DIR, f"perceptron_{gate_name}.pt")
    torch.save(model.state_dict(), weights_path)

    w = model[0].weight.data.numpy().flatten().tolist()   # [w1, w2] from the Linear layer
    b = model[0].bias.data.item()                          # bias from the Linear layer

    meta = {
        "gate": gate_name,
        "weights": w,
        "bias": b,
        "final_loss": loss.item(),
        "final_predictions": predictions.flatten().tolist(),
        "epochs": epochs,
        "learning_rate": lr,
    }
    meta_path = os.path.join(MODELS_DIR, f"perceptron_{gate_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{gate_name}] Saved model weights to: {weights_path}")
    print(f"[{gate_name}] Saved metadata to: {meta_path}")
    print("-" * 60)

    return model, meta


if __name__ == "__main__":
    # Train and save a perceptron for each gate (AND, then OR).
    for gate_name, data in GATES.items():
        train_and_save(gate_name, data["x"], data["y"], epochs=1000, lr=0.5)
