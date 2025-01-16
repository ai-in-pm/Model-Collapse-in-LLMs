import streamlit as st
import random
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Streamlit interface
st.title("Visual Demonstration of Model Collapse with Mixed Training")
st.subheader("Explore the Effects of Synthetic and Real Data on Model Performance")

# Model selection
models = ["OpenAI", "Anthropic", "Mistral", "Groq", "Gemini", "Cohere", "Emergence"]
selected_models = st.multiselect("Select LLMs to Demonstrate", models, default=models)

# Number of generations
generations = st.slider("Number of Generations (Synthetic Data)", 1, 20, 10)
real_data_epochs = st.slider("Number of Generations (Real Data)", 1, 10, 5)

# Display settings
show_tails = st.checkbox("Highlight Tails (Low-Probability Events)", value=True)

# Initialize a random seed for reproducibility
np.random.seed(42)

def generate_distribution(n, highlight_tails=False):
    """
    Generate a synthetic distribution with increasing model collapse effects.
    """
    mean = random.uniform(-1, 1)
    std_dev = max(0.5 - 0.05 * n, 0.1)  # Decreasing variance with each generation
    x = np.linspace(-5, 5, 1000)
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

    # Highlight tails if enabled
    if highlight_tails:
        y = np.where(abs(x - mean) > 2 * std_dev, y * 1.5, y)
    return x, y, mean, std_dev

def generate_real_data_distribution():
    """
    Generate a stable real data distribution.
    """
    mean = 0
    std_dev = 1  # Constant variance for real data
    x = np.linspace(-5, 5, 1000)
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    return x, y

def calculate_perplexity(y):
    """
    Calculate perplexity as the exponent of the entropy of the distribution.
    """
    y_normalized = y / np.sum(y)
    entropy = -np.sum(y_normalized * np.log(y_normalized + 1e-10))  # Add epsilon to avoid log(0)
    perplexity = np.exp(entropy)
    return perplexity

# Visualization and Metrics
if st.button("Generate Demonstration"):
    st.write("### Model Collapse Demonstration with Mixed Training")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Metrics storage
    perplexity_scores = {model: [] for model in selected_models}
    wasserstein_distances = {model: [] for model in selected_models}

    for model in selected_models:
        x_prev, y_prev, mean_prev, std_prev = generate_distribution(0)

        # Synthetic data training
        for gen in range(generations):
            x, y, mean, std = generate_distribution(gen, highlight_tails=show_tails)

            # Calculate metrics
            perplexity_scores[model].append(calculate_perplexity(y))
            wasserstein_distances[model].append(wasserstein_distance(y_prev, y))

            # Update previous generation
            x_prev, y_prev, mean_prev, std_prev = x, y, mean, std

        # Real data training
        x_real, y_real = generate_real_data_distribution()
        for real_epoch in range(real_data_epochs):
            perplexity_scores[model].append(calculate_perplexity(y_real))
            wasserstein_distances[model].append(wasserstein_distance(y_prev, y_real))
            x_prev, y_prev = x_real, y_real

            # Plot only for the final real data stage
            if real_epoch == real_data_epochs - 1:
                ax.plot(x_real, y_real, label=f"{model} - Real Data")

    # Finalize the plot
    ax.set_title("Model Collapse Across Generations with Mixed Training", fontsize=16)
    ax.set_xlabel("Feature Space")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=10)
    st.pyplot(fig)

    # Display Metrics
    st.write("### Metrics Across Generations")
    for model in selected_models:
        st.write(f"**{model}**")
        st.write(f"Perplexity: {perplexity_scores[model]}")
        st.write(f"Wasserstein Distance: {wasserstein_distances[model]}")

    # Plot Perplexity and Wasserstein Distance
    for model in selected_models:
        fig_perp, ax_perp = plt.subplots(figsize=(6, 4))
        ax_perp.plot(range(1, len(perplexity_scores[model]) + 1), perplexity_scores[model], marker="o")
        ax_perp.set_title(f"{model} - Perplexity Across Generations")
        ax_perp.set_xlabel("Generation")
        ax_perp.set_ylabel("Perplexity")
        st.pyplot(fig_perp)

        fig_wass, ax_wass = plt.subplots(figsize=(6, 4))
        ax_wass.plot(range(1, len(wasserstein_distances[model]) + 1), wasserstein_distances[model], marker="o", color="orange")
        ax_wass.set_title(f"{model} - Wasserstein Distance Across Generations")
        ax_wass.set_xlabel("Generation")
        ax_wass.set_ylabel("Wasserstein Distance")
        st.pyplot(fig_wass)

# Explanation
st.write("""
### Key Observations
- **Synthetic Data Training**: Demonstrates the progression of model collapse, with loss of variance and low-probability events.
- **Real Data Recovery**: Shows how training on real data can partially restore the model's representation of the original distribution.

### Key Metrics
- **Perplexity**: Measures how well the model predicts a sequence. Lower perplexity indicates better prediction.
- **Wasserstein Distance**: Quantifies the divergence between distributions. Smaller distances indicate better similarity to the original data.

These metrics and visualizations highlight the importance of including real data in training cycles to mitigate model collapse.
""")
