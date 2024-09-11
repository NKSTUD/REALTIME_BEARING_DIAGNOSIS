from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
from tqdm import tqdm

from charts import plot_features_by_fault, plot_correlation_heatmap
from feature_extractor import all_features, extract_all_features

st.set_page_config(
    page_title="REAL TIME BEARING FAULT DIAGNOSIS",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Data Processing")


@st.cache_data
def read_data(file):
    try:
        if file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            try:
                return pd.read_excel(file, engine='calamine')
            except:
                return pd.read_excel(file)
        else:
            return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def process_data(data_class, file, columns_to_process, features):
    df = read_data(file)[columns_to_process]
    extracted_features = {}
    for col in df.columns:
        extracted_features.update(extract_all_features(df[col]))
    extracted_features["target"] = data_class
    return pd.DataFrame([extracted_features])


data_classes = st.multiselect("What default classes do you have?", ["ball", "combined", "inner", "outer", "safe"])

if data_classes:
    DATA_TO_PROCESS = {}
    for data_class in data_classes:
        uploaded_files = st.file_uploader(f"Choose the files for {data_class}", accept_multiple_files=True,
                                          type=['csv', 'xlsx'])
        if uploaded_files:
            data = read_data(uploaded_files[0])
            st.data_editor(data, disabled=False)
            columns_to_process = st.multiselect(f"Choose the columns to process for {data_class}", data.columns)
            if columns_to_process:
                DATA_TO_PROCESS[data_class] = {"files": uploaded_files, "columns": columns_to_process}

    features = st.multiselect("Choose the features to extract", all_features, default=all_features[:3])

    if st.button("Extract Features"):
        labelled_dataset = pd.DataFrame()
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_files = sum(len(data["files"]) for data in DATA_TO_PROCESS.values())
        processed_files = 0

        with ThreadPoolExecutor() as executor:
            futures = []
            for data_class, data in DATA_TO_PROCESS.items():
                for file in data["files"]:
                    futures.append(executor.submit(process_data, data_class, file, data["columns"], features))

            for future in tqdm(as_completed(futures), total=total_files, desc="Processing files"):
                labelled_dataset = pd.concat([labelled_dataset, future.result()], ignore_index=True)
                processed_files += 1
                progress = processed_files / total_files
                progress_bar.progress(progress)
                status_text.text(f"Processed {processed_files}/{total_files} files")

        st.success("Feature extraction completed!")
        st.write(labelled_dataset)

        # Visualize the labelled dataset

        st.write("Visualizing the labelled dataset...")

        fig = plot_features_by_fault(labelled_dataset, features)
        st.plotly_chart(fig)

        hit_map_fig = plot_correlation_heatmap(labelled_dataset,
                                               [col for col in labelled_dataset.columns if col != "target"])

        st.plotly_chart(hit_map_fig)
