from pathlib import Path
import hashlib
import json
import math
import joblib
import pandas as pd
import plotly.express as px
import sklearn
import streamlit as st

ROOT = Path(__file__).resolve().parent
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
COLORS = dict(zip(CATEGORIES, ["#15803d", "#bef264", "#fde047", "#fb923c", "#dc2626", "#7f1d1d"]))
st.set_page_config(page_title="AirSense", layout="wide")


@st.cache_resource
def load_model(timestamp):

    info = json.loads((ROOT / "model_info.json").read_text(encoding="utf-8"))
    if info["features"] != FEATURES or info["categories"] != CATEGORIES:
        raise ValueError("Incompatible feature schema. Re-run the notebook.")
    if info["sklearn_version"] != sklearn.__version__:
        raise ValueError("Scikit-learn version changed. Re-run the notebook.")
    path = ROOT / "airsense_model.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["model_sha256"]:
        raise ValueError("Model checksum mismatch. Re-run the notebook.")
    model = joblib.load(path)
    if list(model.feature_names_in_) != FEATURES or set(model.classes_) != set(CATEGORIES):
        raise ValueError("Saved model schema does not match the app.")
    return model, info


@st.cache_data
def load_data(timestamp):
    df = pd.read_csv(ROOT / "cleaned_data.csv", parse_dates=["Date"])
    if not set([*FEATURES, "Date", "City", "AQI", "AQI_Bucket"]).issubset(df):
        raise ValueError("Cleaned data is missing columns. Re-run the notebook.")
    return df


def predict_category(values, model):
    if set(values) != set(FEATURES):
        raise ValueError("Provide all six pollutant measurements.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1e6 for v in values.values()):
        raise ValueError("Inputs must be finite numbers between 0 and 1,000,000.")
    frame = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    return str(model.predict(frame)[0]), dict(zip(model.classes_, model.predict_proba(frame)[0].tolist()))


def chart(fig):
    st.plotly_chart(fig, width="stretch")


st.title("AirSense")
st.caption("Air Quality Category Prediction Using Machine Learning")
page = st.sidebar.radio("Explore AirSense", ["Predict", "Explore Data"])


try:
    if page == "Predict":
        model, info = load_model((ROOT / "model_info.json").stat().st_mtime_ns)
        st.header("Predict air quality")
        st.caption("Enter daily concentrations. The model predicts a category, not a numerical AQI.")
        def reset():
            for f in FEATURES:
                st.session_state[f] = float(info["training_ranges"][f]["median"])
            st.session_state.pop("result", None)
        st.button("Load training-median example / reset", on_click=reset)
        with st.form("prediction"):
            cols = st.columns(3)
            values = {}
            for i, f in enumerate(FEATURES):
                unit = "mg/m³" if f == "CO" else "µg/m³"
                values[f] = cols[i % 3].number_input(f"{f} ({unit})", min_value=0.0, max_value=1e6,
                    value=None if f in st.session_state else float(info["training_ranges"][f]["median"]), step=0.1, key=f)
            submitted = st.form_submit_button("Predict Air Quality", type="primary")
        if submitted:
            category, probabilities = predict_category(values, model)
            st.session_state.result = (values, category, probabilities)
        if "result" in st.session_state:
            inputs, category, probabilities = st.session_state.result
            foreground = "#172b1a" if category in ["Satisfactory", "Moderate", "Poor"] else "white"
            st.markdown(f'<div style="background:{COLORS[category]};color:{foreground};padding:24px;border-radius:12px;font-size:28px">Predicted category: {category}</div>', unsafe_allow_html=True)
            ranges = dict(zip(CATEGORIES, ["0–50", "51–100", "101–200", "201–300", "301–400", "401–500"]))
            st.write(f"{category} conventionally corresponds to AQI {ranges[category]}. No numerical AQI has been calculated.")
            st.metric("Model-estimated class probability", f"{probabilities[category]:.1%}")
            st.caption("Uncalibrated probability is not guaranteed correctness or a health-risk estimate.")
            outside = [f for f in FEATURES if not info["training_ranges"][f]["min"] <= inputs[f] <= info["training_ranges"][f]["max"]]
            if outside:
                st.warning("Outside observed training ranges: " + ", ".join(outside))
            st.dataframe(pd.DataFrame({"Category": CATEGORIES, "Probability": [probabilities[c] for c in CATEGORIES]}), hide_index=True)
            record = {**inputs, "predicted_category": category, "model": info["model_name"], **{f"probability_{c}": p for c, p in probabilities.items()}}
            st.download_button("Download prediction CSV", pd.DataFrame([record]).to_csv(index=False).encode("utf-8"), "prediction.csv", "text/csv")
        importance = pd.DataFrame(info["feature_importance"].items(), columns=["Feature", "Importance"])
        chart(px.bar(importance, x="Feature", y="Importance", title="Global feature importance", color_discrete_sequence=["#059669"]))

    elif page == "Explore Data":
        df = load_data((ROOT / "cleaned_data.csv").stat().st_mtime_ns)
        st.header("Explore historical measurements")
        cols = st.columns(3)
        for col, label, value in zip(cols, ["Labeled observations", "Cities", "Categories"], [len(df), df.City.nunique(), df.AQI_Bucket.nunique()]):
            col.metric(label, f"{value:,}")
        cities = st.multiselect("Cities", sorted(df.City.unique()), default=sorted(df.City.unique()))
        dates = st.date_input("Date range", (df.Date.min().date(), df.Date.max().date()))
        categories = st.multiselect("Categories", CATEGORIES, default=CATEGORIES)
        pollutant = st.selectbox("Pollutant", FEATURES)
        if len(dates) != 2:
            st.info("Choose both a start and end date.")
            st.stop()
        filtered = df[df.City.isin(cities) & df.Date.between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1])) & df.AQI_Bucket.isin(categories)]
        if filtered.empty:
            st.info("No observations match these filters.")
            st.stop()
        unit = "mg/m³" if pollutant == "CO" else "µg/m³"
        counts = filtered.AQI_Bucket.value_counts().reindex(CATEGORIES, fill_value=0).rename_axis("Category").reset_index(name="Count")
        chart(px.bar(counts, x="Category", y="Count", color="Category", color_discrete_map=COLORS))
        monthly = filtered.set_index("Date")[["AQI", pollutant]].resample("MS").mean().reset_index()
        chart(px.line(monthly, x="Date", y="AQI", title="Historical monthly mean AQI"))
        chart(px.line(monthly, x="Date", y=pollutant, title=f"Monthly mean {pollutant} ({unit})"))
        st.caption("Multiple cities are pooled; reporting coverage varies over time.")
        chart(px.histogram(filtered, x=pollutant, nbins=50, title=f"{pollutant} distribution ({unit})"))
        st.dataframe(filtered, hide_index=True)
        st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "filtered_data.csv", "text/csv")

except FileNotFoundError:
    st.warning("Training files are not available. Put city_day.csv beside AirSense.ipynb, open the notebook and Run All. It creates the saved model and cleaned data.")
except Exception as exc:
    st.error(f"Unable to load or predict: {exc}")
    st.caption("Re-run AirSense.ipynb to recreate compatible files. Install requirements.txt if a dependency is missing.")
