import pickle
import numpy as np
import streamlit as st

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

SPECIES = ["Setosa", "Versicolor", "Virginica"]

st.title("🌸 Iris Flower Classifier")
st.write("Adjust the sliders and click **Predict** to classify the flower.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 0.2)

# Predict
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.success(f"Predicted Species: **{SPECIES[prediction]}**")
    st.write("### Confidence")
    for i, species in enumerate(SPECIES):
        st.progress(float(probability[i]), text=f"{species}: {probability[i]*100:.1f}%")