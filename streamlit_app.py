"""
Mini Project 12: Binary Pattern Classifier - Streamlit Interface

Loads the perceptron models trained by perceptron_train.py (saved in the
models/ folder as .pt weights + _meta.json files) and lets the user pick
a gate (AND / OR / NAND / NOR / NOT), enter two binary inputs, and see the predicted class,
raw sigmoid probability, and the learned weights/bias.
"""

import os
import json
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from perceptron_train import GATES, MODELS_DIR, train_and_save


def build_model():
    """Same architecture used in training: Linear(2,1) -> Sigmoid."""
    return nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid()
    )


@st.cache_resource
def load_gate(gate_name):
    """
    Loads the trained weights + metadata for a given gate ('AND', 'OR', 'NAND', 'NOR' or 'NOT').
    Cached so the model isn't reloaded from disk on every interaction.
    """
    weights_path = os.path.join(MODELS_DIR, f"perceptron_{gate_name}.pt")
    meta_path = os.path.join(MODELS_DIR, f"perceptron_{gate_name}_meta.json")

    # If the saved files are missing (e.g. models/ was not pushed to GitHub),
    # train the gate now. It is a 4-row truth table, so this takes < 1 second.
    if not (os.path.exists(weights_path) and os.path.exists(meta_path)):
        model, meta = train_and_save(
            gate_name, GATES[gate_name]["x"], GATES[gate_name]["y"],
            epochs=1000, lr=0.5,
        )
        model.eval()
        return model, meta

    model = build_model()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return model, meta



# ----------------------------------------------------------------------
# Linear-separability plots
# ----------------------------------------------------------------------
POINTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
ONE_COLOR, ZERO_COLOR = "#2ecc71", "#e74c3c"


def _draw_points(ax, targets, highlight=None):
    for (a, b), t in zip(POINTS, targets):
        ax.scatter(a, b, s=420, color=ONE_COLOR if t == 1 else ZERO_COLOR,
                   edgecolor="black", linewidth=1.5, zorder=5)
        ax.annotate(str(int(t)), (a, b), ha="center", va="center",
                    color="white", fontweight="bold", fontsize=12, zorder=6)
    if highlight is not None:
        ax.scatter(*highlight, s=1100, facecolor="none", edgecolor="#f39c12",
                   linewidth=3, zorder=4, label="your input")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Input 1 (x1)")
    ax.set_ylabel("Input 2 (x2)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def gate_boundary_figure(gate_name, w1, w2, bias, targets, highlight):
    """Truth-table points + the line the perceptron learned (z = 0, i.e.
    sigmoid output = 0.5). Green side = predicts 1, red side = predicts 0."""
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    gx, gy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    z = w1 * gx + w2 * gy + bias
    ax.contourf(gx, gy, z, levels=[-1e9, 0, 1e9], colors=[ZERO_COLOR, ONE_COLOR],
                alpha=0.13)
    ax.contour(gx, gy, z, levels=[0], colors="black", linewidths=2.5)
    _draw_points(ax, targets, highlight)
    ax.set_title(f"{gate_name}: one straight line separates the 1s from the 0s",
                 fontsize=10)
    ax.text(0.02, 0.02, f"line: {w1:.2f}·x1 + {w2:.2f}·x2 + {bias:.2f} = 0",
            transform=ax.transAxes, fontsize=8, bbox=dict(fc="white", alpha=0.8))
    fig.tight_layout()
    return fig


def xor_figure():
    """XOR truth table with three candidate lines - each gets some points wrong."""
    targets = [0, 1, 1, 0]
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    candidates = [  # (a, b, c): predict 1 when a*x1 + b*x2 > c
        (1, 1, 0.5, "x1 + x2 = 0.5"),
        (1, 1, 1.5, "x1 + x2 = 1.5"),
        (1, 0, 0.5, "x1 = 0.5"),
    ]
    xs = np.linspace(-0.5, 1.5, 50)
    for (a, b, c, label), col in zip(candidates, ["#8e44ad", "#2980b9", "#d35400"]):
        errs = sum(int((a * p + b * q > c) != bool(t))
                   for (p, q), t in zip(POINTS, targets))
        if b == 0:
            ax.axvline(c / a, color=col, ls="--", lw=2, label=f"{label}  ({errs} wrong)")
        else:
            ax.plot(xs, (c - a * xs) / b, color=col, ls="--", lw=2,
                    label=f"{label}  ({errs} wrong)")
    _draw_points(ax, targets)
    ax.set_title("XOR: every straight line gets at least 1 point wrong", fontsize=10)
    ax.legend(loc="lower center", fontsize=8, bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    return fig


# Helper function to generate the ANIMATED neural network SVG diagram

def get_neural_network_svg(x1, x2, w1, w2, b, z, prob, pred):
    """
    Returns an HTML/SVG string visualizing the forward pass with a simple
    CSS-keyframe animation: a signal "pulse" travels from each input node,
    along its weighted edge, into the summation node, then into the sigmoid
    node, and finally the output text fades in. Pure CSS (no JS) so it's
    reliable inside Streamlit's embedded HTML component.

    Layout notes (v2 fix - the previous pass still let the w2 annotation
    collide with the sum node's z-value label):
      - The z-value readout was moved OFF the crowded left column
        entirely. It now lives in the wide open lane between the sum
        node and the sigmoid node (alongside the sigma(z) formula),
        where there is nothing else nearby to collide with.
      - The two weight annotations were pushed further from the sum
        node (they now end well before x=380, while the sum node sits
        at x=430) and are vertically pinned near their own input node
        instead of near the crossing point of the lines, so they can
        never drift into the sum node's own labels.
      - viewBox widened to 860x400 with preserveAspectRatio so nothing
        clips at narrow embed widths.
    """

    x1_color = "#2ecc71" if x1 == 1 else "#3498db"
    x2_color = "#2ecc71" if x2 == 1 else "#3498db"
    prob_color = "#2ecc71" if prob >= 0.5 else "#e74c3c"
    pred_text = "1 (ACTIVE)" if pred == 1 else "0 (INACTIVE)"

    html = f'''
    <style>
        @keyframes pulseNode {{
            0%   {{ transform: scale(1);   filter: brightness(1); }}
            50%  {{ transform: scale(1.15); filter: brightness(1.4); }}
            100% {{ transform: scale(1);   filter: brightness(1); }}
        }}
        @keyframes dashFlow {{
            to {{ stroke-dashoffset: -40; }}
        }}
        @keyframes fadeIn {{
            0%   {{ opacity: 0; transform: translateY(6px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}

        .node-x1  {{ transform-origin: 100px 110px; animation: pulseNode 1.2s ease-in-out 0s   1; }}
        .node-x2  {{ transform-origin: 100px 280px; animation: pulseNode 1.2s ease-in-out 0.15s 1; }}
        .node-bias {{ transform-origin: 430px 60px; animation: pulseNode 1.2s ease-in-out 0.3s  1; }}
        .node-sum {{ transform-origin: 430px 195px; animation: pulseNode 1.2s ease-in-out 0.9s  1; }}
        .node-sig {{ transform-origin: 720px 195px; animation: pulseNode 1.2s ease-in-out 1.6s  1; }}

        .edge-flow {{
            stroke-dasharray: 10 6;
            animation: dashFlow 0.9s linear infinite;
        }}

        .fade-late  {{ opacity: 0; animation: fadeIn 0.6s ease-out 1.9s forwards; }}
        .fade-mid   {{ opacity: 0; animation: fadeIn 0.6s ease-out 1.0s forwards; }}
    </style>

    <svg viewBox="0 0 860 400" preserveAspectRatio="xMidYMid meet"
         xmlns="http://www.w3.org/2000/svg"
         style="width: 100%; height: auto; display: block; background: #f8f9fa; border-radius: 12px; font-family: 'Segoe UI', Arial, sans-serif;">

        <text x="430" y="28" font-size="17" font-weight="bold" fill="#2c3e50" text-anchor="middle">Forward Pass Simulation</text>

        <!-- ====== INPUT LAYER ====== -->
        <circle class="node-x1" cx="100" cy="110" r="30" fill="{x1_color}" stroke="#2c3e50" stroke-width="2"/>
        <text x="100" y="115" font-size="14" font-weight="bold" fill="white" text-anchor="middle">x&#8321;</text>
        <text x="100" y="156" font-size="14" fill="#2c3e50" text-anchor="middle" font-weight="bold">{x1}</text>

        <circle class="node-x2" cx="100" cy="280" r="30" fill="{x2_color}" stroke="#2c3e50" stroke-width="2"/>
        <text x="100" y="285" font-size="14" font-weight="bold" fill="white" text-anchor="middle">x&#8322;</text>
        <text x="100" y="326" font-size="14" fill="#2c3e50" text-anchor="middle" font-weight="bold">{x2}</text>

        <!-- ====== WEIGHT ARROW (x1 -> Sum) - annotation pinned near x1, far from sum ====== -->
        <line class="edge-flow" x1="132" y1="118" x2="400" y2="178" stroke="#34495e" stroke-width="3"/>
        <polygon points="400,178 389,172 391,185" fill="#34495e"/>
        <text x="150" y="92" font-size="13" fill="#e67e22" font-weight="bold">w&#8321; = {w1:.3f}</text>
        <text x="150" y="110" font-size="12" fill="#7f8c8d">({w1:.3f} &#215; {x1} = {w1 * x1:.3f})</text>

        <!-- ====== WEIGHT ARROW (x2 -> Sum) - annotation pinned near x2, far from sum ====== -->
        <line class="edge-flow" x1="132" y1="272" x2="400" y2="212" stroke="#34495e" stroke-width="3"/>
        <polygon points="400,212 389,206 391,219" fill="#34495e"/>
        <text x="150" y="298" font-size="13" fill="#e67e22" font-weight="bold">w&#8322; = {w2:.3f}</text>
        <text x="150" y="316" font-size="12" fill="#7f8c8d">({w2:.3f} &#215; {x2} = {w2 * x2:.3f})</text>

        <!-- ====== BIAS NODE ====== -->
        <circle class="node-bias" cx="430" cy="60" r="22" fill="#9b59b6" stroke="#2c3e50" stroke-width="2"/>
        <text x="430" y="65" font-size="12" font-weight="bold" fill="white" text-anchor="middle">bias</text>
        <text x="430" y="97" font-size="13" fill="#2c3e50" text-anchor="middle" font-weight="bold">{b:.3f}</text>

        <line class="edge-flow" x1="430" y1="104" x2="430" y2="155" stroke="#34495e" stroke-width="2" stroke-dasharray="5,5"/>
        <polygon points="430,155 426,147 434,147" fill="#34495e"/>

        <!-- ====== SUMMATION NODE (Sigma) - kept clear of any other label ====== -->
        <circle class="node-sum" cx="430" cy="195" r="35" fill="#ecf0f1" stroke="#2c3e50" stroke-width="3"/>
        <text x="430" y="203" font-size="24" font-weight="bold" fill="#2c3e50" text-anchor="middle">&#931;</text>
        <text x="430" y="248" font-size="11" fill="#7f8c8d" text-anchor="middle">(sum + bias)</text>

        <!-- ====== OUTPUT ARROW (Sum -> Sigmoid) - z-value + formula live here, open lane ====== -->
        <line class="edge-flow" x1="465" y1="195" x2="685" y2="195" stroke="#34495e" stroke-width="4"/>
        <polygon points="685,195 675,189 675,201" fill="#34495e"/>
        <text class="fade-mid" x="575" y="170" font-size="14" fill="#2c3e50" text-anchor="middle" font-weight="bold">z = {z:.4f}</text>
        <text x="575" y="220" font-size="13" fill="#2980b9" font-weight="bold" text-anchor="middle">&#963;(z) = 1/(1+e^-z)</text>

        <!-- ====== SIGMOID / OUTPUT NODE ====== -->
        <circle class="node-sig" cx="720" cy="195" r="35" fill="{prob_color}" stroke="#2c3e50" stroke-width="3"/>
        <text x="720" y="190" font-size="18" font-weight="bold" fill="white" text-anchor="middle">&#963;</text>
        <text class="fade-late" x="720" y="213" font-size="12" font-weight="bold" fill="white" text-anchor="middle">{prob:.4f}</text>

        <text x="720" y="255" font-size="14" fill="#2c3e50" text-anchor="middle" font-weight="bold">Prediction</text>
        <text class="fade-late" x="720" y="275" font-size="16" fill="{prob_color}" text-anchor="middle" font-weight="bold">{pred_text}</text>

        <!-- ====== LEGEND ====== -->
        <rect x="20" y="365" width="10" height="10" fill="#2ecc71"/>
        <text x="35" y="375" font-size="11" fill="#2c3e50">Active (1)</text>
        <rect x="130" y="365" width="10" height="10" fill="#3498db"/>
        <text x="145" y="375" font-size="11" fill="#2c3e50">Inactive (0)</text>
        <rect x="250" y="365" width="10" height="10" fill="#e67e22"/>
        <text x="265" y="375" font-size="11" fill="#2c3e50">Weighted Edge</text>
        <rect x="410" y="365" width="10" height="10" fill="#9b59b6"/>
        <text x="425" y="375" font-size="11" fill="#2c3e50">Bias</text>
    </svg>
    '''
    return html


# Page layout

st.set_page_config(page_title="Binary Pattern Classifier", page_icon="🔲")
st.title("Binary Pattern Classifier")
st.write(
    "A single-layer perceptron trained on a logic gate's truth table. "
    "Pick a gate, choose two binary inputs, and predict the output."
)

gate_choice = st.selectbox("Choose a logic gate:", list(GATES))

st.info(
    "**Why no XOR?** A single-layer perceptron can't learn XOR, because XOR "
    "isn't linearly separable: no single straight line can split the inputs "
    "(0,0) and (1,1) from (0,1) and (1,0). AND, OR, NAND and NOR are "
    "linearly separable, so one layer is enough. XOR needs at least one "
    "hidden layer."
)
model, meta = load_gate(gate_choice)

st.subheader("Inputs")
col1, col2 = st.columns(2)
with col1:
    x1 = st.radio("Input 1", [0, 1], horizontal=True)
with col2:
    if gate_choice == "NOT":
        x2 = 0
        st.markdown("**Input 2**")
        st.caption("Not used: NOT is a one-input gate, so only Input 1 matters.")
    else:
        x2 = st.radio("Input 2", [0, 1], horizontal=True)

st.subheader("Why one layer is enough: linear separability")
_t = GATES[gate_choice]["y"].flatten().tolist()
_w1, _w2 = meta["weights"]
st.pyplot(gate_boundary_figure(gate_choice, _w1, _w2, meta["bias"], _t, (x1, x2)))
plt.close("all")
st.caption(
    "Each dot is one row of the truth table (green = output 1, red = output 0). "
    "The black line is what the perceptron learned: where its weighted sum "
    "z = 0, so the sigmoid gives exactly 0.5. Everything on the green side is "
    "predicted 1 and everything on the red side is predicted 0. Because one "
    "straight line puts every green dot on one side and every red dot on the "
    "other, the gate is **linearly separable** and a single-layer perceptron "
    "can learn it. The orange ring marks the input you selected."
)

with st.expander("What about XOR? See why one line can never work"):
    st.pyplot(xor_figure())
    plt.close("all")
    st.markdown(
        "XOR outputs 1 only when the inputs **differ**. The two green dots sit "
        "on opposite corners, and so do the two red dots, so no single straight "
        "line can put both greens on one side and both reds on the other. The "
        "best any line can do is get 3 of the 4 rows right (75%). Fixing this "
        "needs a **hidden layer**, which lets the network combine two lines."
    )

if st.button("Predict"):
    x_tensor = torch.tensor([[float(x1), float(x2)]])

    with torch.no_grad():
        probability = model(x_tensor).item()

    predicted_class = 1 if probability >= 0.5 else 0

    st.subheader("Prediction")
    st.metric(label=f"{gate_choice} Gate Output", value=predicted_class)
    st.write(f"Raw sigmoid probability: **{probability:.4f}**")

    st.progress(min(max(probability, 0.0), 1.0))

    
    # Neural network simulation (forward pass)

    st.divider()
    st.subheader(" Neural Network Simulation (Forward Pass)")

    w1, w2 = meta["weights"]
    bias = meta["bias"]

    # Weighted sum
    z = (w1 * x1) + (w2 * x2) + bias

    # ---------- Row 1 ----------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1. Weighted Sum")
        st.latex(rf"""
        \begin{{aligned}}
        z &= ({w1:.2f}\times{x1}) \\
        &+ ({w2:.2f}\times{x2}) \\
        &+ ({bias:.2f}) \\
        &= {z:.2f}
        \end{{aligned}}
        """)

    with col2:
        st.markdown("### 2. Sigmoid Activation")
        st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}}")
        st.latex(rf"\sigma({z:.2f})={probability:.4f}")

    st.divider()

    # ---------- Decision Rule ----------
    st.markdown("### 3. Decision Rule")

    rule_col1, rule_col2 = st.columns([2,1])

    with rule_col1:
        st.markdown("""
    - **If** σ(z) ≥ **0.5** → **Output = 1**
    - **If** σ(z) < **0.5** → **Output = 0**
    """)

    with rule_col2:
        st.metric("Prediction", predicted_class)

    st.markdown("### Network Architecture & Data Flow")
    st.caption(
        "Watch the signal travel from the input nodes through the weighted sum "
        "and sigmoid activation to produce the final prediction."
    )

    svg_diagram = get_neural_network_svg(
        x1, x2, w1, w2, bias, z, probability, predicted_class
    )

    st.components.v1.html(svg_diagram, height=380)

st.divider()


# Learned parameters

st.subheader("Learned Parameters")
w1, w2 = meta["weights"]
bias = meta["bias"]

param_col1, param_col2, param_col3 = st.columns(3)
param_col1.metric("Weight 1 (w1)", f"{w1:.4f}")
param_col2.metric("Weight 2 (w2)", f"{w2:.4f}")
param_col3.metric("Bias (b)", f"{bias:.4f}")

with st.expander("Training details"):
    st.write(f"Epochs: {meta['epochs']}")
    st.write(f"Learning rate: {meta['learning_rate']}")
    st.write(f"Final training loss: {meta['final_loss']:.6f}")
    st.write("Final predictions on the full truth table:")
    st.write([round(p, 4) for p in meta["final_predictions"]])
