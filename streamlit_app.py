"""
Mini Project 12: Binary Pattern Classifier - Streamlit Interface

Loads the perceptron models trained by perceptron_train.py (saved in the
models/ folder as .pt weights + _meta.json files) and lets the user pick
a gate (AND / OR), enter two binary inputs, and see the predicted class,
raw sigmoid probability, and the learned weights/bias.
"""

import os
import json
import torch
import torch.nn as nn
import streamlit as st

MODELS_DIR = "models"


def build_model():
    """Same architecture used in training: Linear(2,1) -> Sigmoid."""
    return nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid()
    )


@st.cache_resource
def load_gate(gate_name):
    """
    Loads the trained weights + metadata for a given gate ('AND' or 'OR').
    Cached so the model isn't reloaded from disk on every interaction.
    """
    weights_path = os.path.join(MODELS_DIR, f"perceptron_{gate_name}.pt")
    meta_path = os.path.join(MODELS_DIR, f"perceptron_{gate_name}_meta.json")

    model = build_model()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return model, meta



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

gate_choice = st.selectbox("Choose a logic gate:", ["AND", "OR"])
model, meta = load_gate(gate_choice)

st.subheader("Inputs")
col1, col2 = st.columns(2)
with col1:
    x1 = st.radio("Input 1", [0, 1], horizontal=True)
with col2:
    x2 = st.radio("Input 2", [0, 1], horizontal=True)

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