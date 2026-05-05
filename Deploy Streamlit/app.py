import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os


@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "Fish.csv")
    df = pd.read_csv(file_path)
    return df


@st.cache_data
def train_model(df):
    X = df.drop("Species", axis=1)
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return model, accuracy


def main():
    st.set_page_config(
        page_title="Klasifikasi Ikan dengan Streamlit", layout="centered"
    )
    st.title("🐟 Aplikasi Klasifikasi Spesies Ikan")
    st.write("Masukkan ukuran fisik ikan untuk memprediksi spesiesnya:")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(
            "File 'Fish.csv' tidak ditemukan! Pastikan file berada di folder yang sama dengan app.py."
        )
        return

    model, accuracy = train_model(df)

    col1, col2 = st.columns(2)

    with col1:
        weight = st.slider(
            "Weight (Gram)",
            float(df["Weight"].min()),
            float(df["Weight"].max()),
            float(df["Weight"].mean()),
        )
        length1 = st.slider(
            "Length1 / Standard Length (cm)",
            float(df["Length1"].min()),
            float(df["Length1"].max()),
            float(df["Length1"].mean()),
        )
        length2 = st.slider(
            "Length2 / Fork Length (cm)",
            float(df["Length2"].min()),
            float(df["Length2"].max()),
            float(df["Length2"].mean()),
        )

    with col2:
        length3 = st.slider(
            "Length3 / Total Length (cm)",
            float(df["Length3"].min()),
            float(df["Length3"].max()),
            float(df["Length3"].mean()),
        )
        height = st.slider(
            "Height (cm)",
            float(df["Height"].min()),
            float(df["Height"].max()),
            float(df["Height"].mean()),
        )
        width = st.slider(
            "Width (cm)",
            float(df["Width"].min()),
            float(df["Width"].max()),
            float(df["Width"].mean()),
        )

    if st.button("Prediksi"):
        input_data = pd.DataFrame(
            [[weight, length1, length2, length3, height, width]],
            columns=["Weight", "Length1", "Length2", "Length3", "Height", "Width"],
        )

        prediction = model.predict(input_data)[0]
        st.success(f"🎣 Prediksi spesies ikan: **{prediction}**")
        st.info(f"Akurasi model pada data uji: {accuracy * 100:.2f}%")

    if st.checkbox("Tampilkan Dataset Fish"):
        st.dataframe(df)


if __name__ == "__main__":
    main()
