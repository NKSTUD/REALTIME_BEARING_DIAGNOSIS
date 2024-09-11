import logging
import os
import time
from os.path import expanduser
from pathlib import Path

import pandas as pd
import streamlit as st

from charts import plot_signal, plot_envelope_fft, model_predictions, plot_predictions
from data_processor import get_data
from feature_extractor import calculate_features, Signal, all_features, extract_model_features
from sensor_config import format_sensors_for_get_data, sensor_configuration

st.set_page_config(
    page_title="REAL TIME BEARING FAULT DIAGNOSIS",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("  :green[REAL TIME] BEARING :red[FAULT] DIAGNOSIS IN MOROCCO HIGH SPEED TRAINS")

if "stream" not in st.session_state:
    st.session_state.stream = False

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()

if "previous_features" not in st.session_state:
    st.session_state.previous_features = [{}, {}, {}, {}]

if "selected_sensors" not in st.session_state:
    st.session_state.selected_sensors = {}

if "data_acquisition" not in st.session_state:
    st.session_state.data_acquisition = False

if "file_index" not in st.session_state:
    st.session_state.file_index = 0

if "feature_data" not in st.session_state:
    st.session_state.feature_data = pd.DataFrame()

if "is_simulation" not in st.session_state:
    st.session_state.is_simulation = False


# Functions
def toggle_stream():
    st.session_state.stream = not st.session_state.stream
    if st.session_state.stream:
        print("Start streaming...")
    else:
        print("Stop streaming...")


def toggle_data_acquisition():
    st.session_state.data_acquisition = not st.session_state.data_acquisition
    if st.session_state.data_acquisition:
        print("Start data acquisition...")
    else:
        print("Stop data acquisition...")

    # if st.session_state.feature_data is not None:
    #     st.session_state.feature_data = pd.DataFrame()


def save_file(data: pd.DataFrame, file_path: str):
    home_dir = expanduser("~")
    default_output_folder = os.path.join(home_dir, "Desktop", "data_output")
    os.makedirs(default_output_folder, exist_ok=True)
    data.to_csv(os.path.join(default_output_folder, f"{Path(file_path).stem}.csv"), index=False)


def get_all_models():
    """return models in the models folder"""

    models = []
    for model in os.listdir("models"):
        if model.endswith(".pkl"):
            models.append(model)

    return models


# UI
simulation = st.sidebar.checkbox("Simulation Mode", key="simulation")
button_label = "Stop Streaming" if st.session_state.stream else "Start Streaming"
st.sidebar.button(button_label, on_click=toggle_stream)

models = st.sidebar.selectbox("Select Model", get_all_models(), key="model")
with st.sidebar.expander("Data Acquisition Settings", expanded=False):
    acquisition_type = st.multiselect("Acquisition Type", ["raw", "features"], default=["raw", "features"],
                                      key="acquisition_type")
    sampling_rate = st.number_input("Sampling Rate (Hz)", value=25600, min_value=1)
    number_of_samples = st.number_input("Number of Samples", value=25600, min_value=1)
    number_of_files = st.number_input("Number of Files", value=500, min_value=1)
    output_folder = st.text_input("Output Folder", value="data_output")

    button_label = "Stop Data Acquisition" if st.session_state.data_acquisition else "Start Data Acquisition"
    st.button(button_label, on_click=toggle_data_acquisition)

# Sensor Configuration Section
with st.expander("Sensor Configuration", expanded=False):
    selected_sensors = sensor_configuration()
    st.session_state.selected_sensors = selected_sensors

    # Display the configured sensors
    formatted_sensors = format_sensors_for_get_data(selected_sensors)
    st.write(formatted_sensors)

# Create placeholders
placeholders = {
    "charts": [],
    "features": [],
    "data": None,
    "features_data": None,
    "model_predictions": [],
}

# SIGNALS DISPLAY
containers = [st.container(border=True) for _ in range(len(formatted_sensors) + 3)]

for i, container in enumerate(containers[1:-2]):
    with container:
        col1, col2 = st.columns([0.7, 0.3], gap="medium")
        with col1:
            signal_chart = st.empty()
            fft_chart = st.empty()
        with col2:
            features = {
                "crest_factor": st.empty(),
                "kurtosis": st.empty(),
                "skewness": st.empty(),
                "rms": st.empty(),
            }

            model_predictions_placeholder = {
                "chart": st.empty(),
                "best_class": st.empty(),
            }

        placeholders["model_predictions"].append(model_predictions_placeholder)

        placeholders["charts"].append((signal_chart, fft_chart))
        placeholders["features"].append(features)

with containers[-2]:
    col1, col2 = st.columns([0.3, 0.7], gap="medium")
    with col1:
        number_of_file_placeholder = st.empty()

    with col2:
        data_display = st.empty()
        placeholders["data"] = (number_of_file_placeholder, data_display)

with containers[-1]:
    features_data_placeholder = st.empty()
    placeholders["features_data"] = features_data_placeholder

# features_data = pd.DataFrame()


if "features_data" not in st.session_state:
    st.session_state.features_data = pd.DataFrame()

selected_features = st.multiselect(
    "Sélectionnez les features à extraire",
    all_features,
    default=["rms", "mean", "kurtosis", "skewness"], label_visibility="collapsed"
)

# Main loop
logging.basicConfig(level=logging.INFO)

while st.session_state.stream:
    logging.info("Starting new iteration of data acquisition loop")

    if simulation:
        st.session_state.is_simulation = True
    else:
        st.session_state.is_simulation = False

    # Get data
    sensors = format_sensors_for_get_data(st.session_state.selected_sensors)
    data = st.session_state.data = get_data(is_started=st.session_state.stream,
                                            is_simulation=st.session_state.is_simulation,
                                            num_samples=number_of_samples,
                                            sampling_rate=sampling_rate,
                                            sensors=sensors)

    logging.info(f"Acquired data shape: {data.shape}")

    if data.empty:
        logging.warning("Received empty data from get_data()")
        st.write("No data available. Skipping this iteration.")
        time.sleep(1.5)  # Wait a bit before trying again
        continue

    # Update placeholders
    for i, column in enumerate(data.columns):
        column_data = data[column]
        logging.info(f"Processing column {column} with data shape: {column_data.shape}")

        features = calculate_features(column_data)
        if i >= len(st.session_state.previous_features):
            st.session_state.previous_features.append({})

        # Update signal chart
        signal_fig = plot_signal(data[column], title=f"{column} Time Signal")
        placeholders["charts"][i][0].plotly_chart(signal_fig, use_container_width=True)

        # Update FFT chart
        if column_data is not None and len(column_data) > 0:
            fft_fig = plot_envelope_fft(x=column_data, fs=sampling_rate, title="FFT of Signal Envelope")
            placeholders["charts"][i][1].plotly_chart(fft_fig, use_container_width=True)
        else:
            logging.warning(f"Empty column data for {column}")
            st.write(f"No data available for FFT calculation for {column}.")
            st.write(f"Column data: {column_data}")

        # Update features
        for feature, value in features.items():
            previous_value = st.session_state.previous_features[i].get(feature, value)
            delta = value - previous_value
            placeholders["features"][i][feature].metric(
                feature.capitalize(),
                f"{value:.4f}",
                delta=f"{delta:.4f}"
            )
            st.session_state.previous_features[i][feature] = value

        # Update model predictions

        features = extract_model_features(column_data)

        if features:
            # features = {key: value for key, value in features.items() if key in model_features}
            probabilities = model_predictions(features)
            placeholders["model_predictions"][i]["chart"].plotly_chart(plot_predictions(probabilities),
                                                                       use_container_width=True)
            best_class = max(probabilities, key=probabilities.get)

            placeholders["model_predictions"][i]["best_class"].metric("Predicted Class", best_class,
                                                                      f"{probabilities[best_class] * 100:.2f}%")

        if st.session_state.data_acquisition:
            if "features" in acquisition_type:
                signal = Signal(column_data, sampling_rate)
                features = signal.extract_features()
                final_features = {key: value for key, value in features.items() if key in selected_features}
                feature_data = pd.DataFrame(final_features, index=[0])
                st.session_state.feature_data = pd.concat([st.session_state.feature_data, feature_data],
                                                          ignore_index=True)
                features_data_placeholder.write(st.session_state.feature_data.style.highlight_max(axis=0))

    # Data acquisition logic
    if st.session_state.data_acquisition:
        if "raw" in acquisition_type:
            st.session_state.file_index += 1
            save_file(st.session_state.data, os.path.join(output_folder, f"file_{st.session_state.file_index}"))
            placeholders["data"][0].metric(f"Number of Saved File", st.session_state.file_index)
            placeholders["data"][1].write(st.session_state.data)

        # Stop acquisition after the specified number of files
        if st.session_state.file_index >= number_of_files:
            st.session_state.data_acquisition = False
            logging.info("Data acquisition completed")

    placeholders["data"][1].write(st.session_state.data.style.highlight_max(axis=0))

    # Add a small delay at the end of each iteration
    time.sleep(0.1)

logging.info("Exited data acquisition loop")

# Display the final features data

if not st.session_state.stream:
    features_data_placeholder.write(st.session_state.features_data)
    placeholders["data"][1].write(st.session_state.data)

if not st.session_state.data_acquisition:
    features_data_placeholder.write(
        st.session_state.feature_data.style.highlight_max(axis=0) or "No features data available.")
