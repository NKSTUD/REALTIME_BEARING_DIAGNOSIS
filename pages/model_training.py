import joblib
import pandas as pd
import streamlit as st
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, \
    Perceptron
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from pages.data_processing import read_data

st.title("Model Training")

uploaded_file = st.file_uploader("Choose a file of data", accept_multiple_files=True, type=['csv', 'xlsx'])
if uploaded_file:

    if len(uploaded_file) > 1:
        # concatenate all the dataframes
        data = pd.concat([read_data(file) for file in uploaded_file], ignore_index=True)

        st.data_editor(data, disabled=False)
    else:
        data = read_data(uploaded_file[0])
        st.data_editor(data, disabled=False)

    # st.write("Choose the target column")

    target_column = st.selectbox("Choose the target column", data.columns)

    # st.write("Choose the models to train")

    model_options = {
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Bagging": BaggingClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "Logistic Regression": OneVsRestClassifier(LogisticRegression()),  # OvR for multiclass
        "Ridge Classifier": RidgeClassifier(),
        "SGD Classifier": OneVsRestClassifier(SGDClassifier()),  # OvR for multiclass
        "Passive Aggressive": PassiveAggressiveClassifier(),
        "Perceptron": Perceptron(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "SVM": SVC(decision_function_shape='ovo'),  # OvO strategy
        "NuSVC": NuSVC(decision_function_shape='ovo'),  # OvO strategy
        "Linear SVC": LinearSVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Extra Tree": ExtraTreeClassifier(),
        "Gaussian NB": GaussianNB(),
        "Multinomial NB": MultinomialNB(),
        "Bernoulli NB": BernoulliNB(),
        "MLP Classifier": MLPClassifier(),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis()
    }

    selected_models = st.multiselect("Select models to include in the pipeline", list(model_options.keys()))

    if selected_models:
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2)

        if st.button("Train Models"):
            if not selected_models:
                st.error("Please select at least one model.")
            else:
                # Split data
                X = data.drop(target_column, axis=1)
                y = data[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                    random_state=42)

                # Train models
                results = {}
                progress_bar = st.progress(0)
                for i, model_name in enumerate(selected_models):
                    try:
                        model = model_options[model_name]
                        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
                        pipeline.fit(X_train, y_train)

                        # Predict and evaluate
                        y_pred = pipeline.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        confusion = confusion_matrix(y_test, y_pred)

                        # Store results
                        results[model_name] = {
                            "pipeline": pipeline,
                            "confusion_matrix": confusion,
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                        }

                    except Exception as e:
                        # Display the error message
                        st.error(f"An error occurred while training {model_name}: {e}")

                    # Update progress bar
                    progress_bar.progress((i + 1) / len(selected_models))
                st.success("Training completed!")

                st.header("Comparing models Results")
                model_results = pd.DataFrame.from_dict(results, orient='index')
                model_results = model_results[['accuracy', 'precision', 'recall', 'f1']]
                st.write(model_results.style.highlight_max(axis=0))

                # Display results
                st.header("Training Results")
                for model_name, result in results.items():
                    st.subheader(model_name)
                    st.write("Confusion Matrix:")
                    st.write(pd.DataFrame(result["confusion_matrix"],
                                          columns=[f"Predicted {c}" for c in result["pipeline"].classes_],
                                          index=[f"Actual {c}" for c in result["pipeline"].classes_]))

                    # Offer to download the model
                    if st.button(f"Deploy {model_name}", key=model_name, help=f"Deploy the {model_name} model"):
                        model_filename = f"{model_name}_model.pkl"
                        joblib.dump(result["pipeline"], f"{model_name}")
                        st.success(f"{model_name} deployed successfully!")
                        with open(model_filename, "rb") as file:
                            st.download_button(f"Download {model_name} model", file, file_name=model_filename)




