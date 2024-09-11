from pathlib import Path
from typing import List

import altair as alt
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.fft import fft
from scipy.signal import hilbert


def plot_time_signal(data: pd.DataFrame, title: str = "Time Signal", xlabel: str = "Time", ylabel: str = "Amplitude"):
    """Plot time signal using Altair with xlim and ylim."""
    base_chart = alt.Chart(data.reset_index()).mark_line().encode(
        x=alt.X('Time', title=xlabel),
        y=alt.Y('Amplitude', title=ylabel)
    ).properties(
        title=title,
        width=400,
        height=300
    )
    return base_chart


def plot_signal(data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name=title))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Amplitude")
    return fig


def plot_fft(data, title):
    # Convertir les données en tableau numpy si ce n'est pas déjà le cas
    data_array = np.array(data)
    fft_result = fft(data_array)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.abs(fft_result), mode='lines', name=f"{title} FFT"))
    fig.update_layout(title=f"{title} FFT", xaxis_title="Frequency", yaxis_title="Magnitude")

    return fig


def my_envelope_fft(x, fs):
    """Plot the FFT of the signal envelope"""

    if len(x) == 0:
        raise ValueError("The input signal x is empty.")

    NN = len(x)
    ff = np.linspace(0, fs, NN + 1)
    ff = ff[:NN]

    x_envelope = np.abs(hilbert(x))

    amp = np.abs(np.fft.fft(x_envelope - np.mean(x_envelope)) ** 2) / NN / 2
    amp = amp[:NN // 2]
    ff = ff[:NN // 2]

    fft_data = pd.DataFrame({"Frequency": ff, "Amplitude": amp})
    fft_data.set_index("Frequency", inplace=True)

    return fft_data


def plot_envelope_fft(x, fs, title="FFT of Signal Envelope"):
    """
    Compute and plot the FFT of the signal envelope using Plotly.

    Parameters:
    - x: The input signal (1D array or list).
    - fs: The sampling frequency of the signal.
    """

    fft_data = my_envelope_fft(x, fs)

    # Create a Plotly figure
    fig = go.Figure()

    # Add the FFT data as a line plot
    fig.add_trace(go.Scatter(x=fft_data.index, y=fft_data['Amplitude'], mode='lines', name='FFT Envelope'))

    fig.update_layout(
        title="FFT of Signal Envelope",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        template="plotly_white"
    )

    return fig


def plot_predictions(probabilities):
    """Plot model predictions using Plotly graph objects."""

    # Convert the probabilities dictionary to a pandas DataFrame
    df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])

    # Create a pie chart using Plotly graph_objects
    fig = go.Figure(
        data=[go.Pie(labels=df['Class'], values=df['Probability'])],
        layout_title_text='Model Predictions'
    )

    return fig


pwd = Path(__file__).parent.absolute().__str__()

print(pwd)
print(type(pwd))


def load_model(model_path: str = r"models/svm_pipeline.pkl"):
    """Load a trained model."""
    model = joblib.load(model_path)
    return model


def model_predictions(features, model_path=r"models/rf_pipeline.pkl"):
    """Return predicted classes and their probabilities."""

    model = load_model(model_path)

    features_df = pd.DataFrame([features])

    # Predict probabilities
    probabilities = model.predict_proba(features_df)[0]

    # Define labels
    labels = ['ball', 'combined', 'inner', 'outer', 'safe']

    # Create a dictionary of classes and their probabilities
    class_probabilities = {label: prob for label, prob in zip(labels, probabilities)}

    # Sort the dictionary by probability in descending order
    sorted_class_probabilities = dict(sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True))

    return sorted_class_probabilities


def plot_features_by_fault(df: pd.DataFrame, features: List[str], fault_column: str = 'target'):
    """
    Plot multiple features by fault type using a grouped bar plot with Plotly.

    :param df: DataFrame containing the data
    :param features: List of feature names to plot
    :param fault_column: Name of the column containing fault types
    """
    # Calculer la moyenne de chaque feature pour chaque type de faute
    df_mean_features = df.groupby(fault_column)[features].mean().reset_index()

    # Créer la figure
    fig = go.Figure()

    # Ajouter une trace pour chaque feature
    for feature in features:
        fig.add_trace(go.Bar(
            x=df_mean_features[fault_column],
            y=df_mean_features[feature],
            name=feature.replace('_', ' ').capitalize(),
            text=df_mean_features[feature].round(2),
            textposition='outside'
        ))

    # Mise à jour de la mise en page pour un graphique en pleine page
    fig.update_layout(
        barmode='group',
        title={
            'text': 'Moyennes des Features par Type de Faute',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis_title={'text': 'Type de Faute', 'font': {'size': 18}},
        yaxis_title={'text': 'Valeur Moyenne', 'font': {'size': 18}},
        legend_title={'text': 'Features', 'font': {'size': 18}},
        font=dict(size=14),
        height=600,  # Hauteur en pixels
        width=1000,  # Largeur en pixels
        margin=dict(l=50, r=50, t=100, b=100),  # Marges pour s'assurer que tout est visible
    )

    # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
    fig.update_xaxes(tickangle=45, tickfont=dict(size=14))

    # Ajustement de la taille des légendes
    fig.update_legends(font=dict(size=14))

    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: List[str]):
    """
    Create an interactive correlation heatmap for the given features using Plotly.

    :param df: DataFrame containing the data
    :param features: List of feature names to include in the heatmap
    """
    # Calculer la matrice de corrélation
    corr_matrix = df[features].corr()

    # Créer la heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        zmin=-1, zmax=1,
        colorscale='RdBu',
        colorbar=dict(title='Corrélation')
    ))

    # Mise à jour de la mise en page
    fig.update_layout(
        title='Heatmap de Corrélation des Features',
        xaxis_title='Features',
        yaxis_title='Features',
        width=800,
        height=800,
    )

    # Ajout des valeurs de corrélation sur la heatmap
    annotations = []
    for i, row in enumerate(corr_matrix.values):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=corr_matrix.columns[j],
                    y=corr_matrix.columns[i],
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(color='white' if abs(value) > 0.5 else 'black')
                )
            )
    fig.update_layout(annotations=annotations)

    return fig
