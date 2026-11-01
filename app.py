
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

st.set_page_config(
    page_title="AutoValue | Car Price Predictor",
    page_icon="🚘",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "car_price_model_improved.pkl"
DATA_FILE = BASE_DIR / "car data.csv"

# ---------- PREMIUM DARK THEME ----------
st.markdown("""
<style>
.stApp {
    background:
        linear-gradient(
            90deg,
            rgba(3, 10, 23, 0.88),
            rgba(3, 10, 23, 0.55)
        ),
        url("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=2000&q=90");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #f1f5f9;
}
}
.block-container {
    max-width: 1350px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
p, label, .stMarkdown {
    color: #dbe4f0;
}
div[data-testid="stMetric"] {
    background: #172033;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 16px;
}
div[data-testid="stMetricValue"] {
    color: #34d399;
}
div.stButton > button, div.stDownloadButton > button {
    background: #10b981;
    color: #06121b;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    background: #6ee7b7;
    color: #06121b;
}
section[data-testid="stSidebar"] {
    background: #0b1220;
}
div[data-testid="stTabs"] button {
    font-weight: 600;
}
/* PREMIUM GLASS EFFECT */

div[data-testid="stForm"] {
    background: rgba(8, 22, 42, 0.82);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(0, 230, 170, 0.55);
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 0 30px rgba(0, 220, 170, 0.15);
}

/* NEON GREEN BUTTONS */

div.stButton > button,
div.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #00e6a8, #00b98b);
    color: #03121b;
    border: none;
    border-radius: 14px;
    font-weight: 800;
    min-height: 45px;
    transition: 0.3s ease;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 0 22px rgba(0, 255, 180, 0.55);
    transform: translateY(-2px);
}

/* GLASS METRIC CARDS */

div[data-testid="stMetric"] {
    background: rgba(15, 35, 55, 0.80);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 220, 170, 0.30);
    border-radius: 16px;
    padding: 18px;
}

/* INPUT FIELDS */

.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    background-color: #10243b;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODEL AND DATA ----------
if not MODEL_FILE.exists():
    st.error("car_price_model_improved.pkl nahi mili. Use app.py ke same folder mein rakho.")
    st.stop()

if not DATA_FILE.exists():
    st.error("car data.csv nahi mili. Use app.py ke same folder mein rakho.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_FILE)


try:
    model_data = load_model()
    model = model_data["model"]
    df = load_dataset()
except Exception as e:
    st.error("Model ya dataset load nahi hua.")
    st.code(str(e))
    st.stop()


required = [
    "name", "year", "selling_price", "km_driven",
    "fuel", "seller_type", "transmission", "owner"
]
missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"Dataset mein ye columns missing hain: {missing}")
    st.stop()

df = df.dropna(subset=required).copy()
df["name"] = df["name"].astype(str)
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["km_driven"] = pd.to_numeric(df["km_driven"], errors="coerce")
df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
df = df.dropna(subset=["year", "km_driven", "selling_price"])

if df.empty:
    st.error("Dataset mein valid car records nahi mile.")
    st.stop()

current_year = datetime.now().year
min_year = int(df["year"].min())
max_year = min(int(df["year"].max()), current_year)

car_names = sorted(df["name"].unique().tolist())
fuel_options = sorted(df["fuel"].astype(str).unique().tolist())
transmission_options = sorted(df["transmission"].astype(str).unique().tolist())
seller_options = sorted(df["seller_type"].astype(str).unique().tolist())
owner_options = sorted(df["owner"].astype(str).unique().tolist())

if not car_names or not fuel_options or not transmission_options:
    st.error("Dataset mein car options available nahi hain.")
    st.stop()


# ---------- PREDICTION FUNCTION ----------
def predict_car(name, year, km_driven, fuel, transmission, seller_type, owner):
    # Keep these columns aligned with the trained model.
    input_data = pd.DataFrame([{
        "brand": str(name).split()[0],
        "vehicle_age": max(0, current_year - int(year)),
        "km_driven": int(km_driven),
        "fuel": fuel,
        "transmission": transmission,
        "seller_type": seller_type,
        "owner": owner
    }])

    price = float(model.predict(input_data)[0])
    return max(0, price), input_data.iloc[0].to_dict()


def car_form(prefix, title):
    st.markdown(f"#### {title}")

    name = st.selectbox(
        "Car model",
        car_names,
        key=f"{prefix}_name"
    )

    year = st.number_input(
        "Manufacturing year",
        min_value=min_year,
        max_value=max_year,
        value=max(min_year, min(max_year, current_year - 5)),
        step=1,
        key=f"{prefix}_year"
    )

    km_driven = st.number_input(
        "Kilometres driven",
        min_value=0,
        max_value=2_000_000,
        value=30_000,
        step=1_000,
        key=f"{prefix}_km"
    )

    fuel = st.selectbox(
        "Fuel type",
        fuel_options,
        key=f"{prefix}_fuel"
    )

    transmission = st.selectbox(
        "Transmission",
        transmission_options,
        key=f"{prefix}_transmission"
    )

    seller_type = st.selectbox(
        "Seller type",
        seller_options,
        key=f"{prefix}_seller"
    )

    owner = st.selectbox(
        "Previous owner",
        owner_options,
        key=f"{prefix}_owner"
    )

    st.caption(f"Vehicle age: {max(0, current_year - int(year))} years")

    return {
        "name": name,
        "year": int(year),
        "km_driven": int(km_driven),
        "fuel": fuel,
        "transmission": transmission,
        "seller_type": seller_type,
        "owner": owner
    }


def add_history(car, price):
    st.session_state.history.append({
        "Date & Time": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d-%m-%Y %I:%M:%S %p"),
        "Car Model": car["name"],
        "Year": car["year"],
        "Vehicle Age": max(0, current_year - car["year"]),
        "Kilometres Driven": car["km_driven"],
        "Fuel": car["fuel"],
        "Transmission": car["transmission"],
        "Seller Type": car["seller_type"],
        "Owner": car["owner"],
        "Estimated Price (₹)": round(price, 2)
    })


# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_car" not in st.session_state:
    st.session_state.last_car = None


# ---------- HEADER ----------
# ---------- PREMIUM HERO SECTION ----------

st.markdown("""
<div style="
    padding: 55px 35px;
    border-radius: 25px;
    background: linear-gradient(110deg, rgba(3,12,27,0.95), rgba(4,35,43,0.75));
    border: 1px solid rgba(0,230,170,0.35);
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
">
    <div style="color:#00e6a8;font-size:13px;font-weight:800;letter-spacing:4px;">
        SMART • FAST • DATA-DRIVEN
    </div>
    <h1 style="color:white;font-size:52px;font-weight:900;">
        AutoValue
    </h1>
    <h2 style="color:#00e6a8;font-size:28px;">
        Smart Car Price Predictor
    </h2>
    <p style="color:#dbeafe;font-size:17px;">
        Know your car's real value. Make smarter decisions.
        Powered by Machine Learning.
    </p>
    <div style="color:#a7f3d0;font-size:14px;margin-top:25px;">
        ✦ AI-Powered Predictions &nbsp;&nbsp;
        ✦ Smart Car Comparison &nbsp;&nbsp;
        ✦ Data Analytics
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")
st.divider()


# ---------- SIDEBAR ----------
st.sidebar.title("🚘 AutoValue")
st.sidebar.caption("Your used-car analysis dashboard")
st.sidebar.metric("Cars in dataset", f"{len(df):,}")
st.sidebar.metric("Average listed price", f"₹{df['selling_price'].mean():,.0f}")
st.sidebar.metric("Model", model_data.get("model_name", "Machine Learning"))
st.sidebar.divider()
st.sidebar.info(
    "Predictions are estimates. Actual resale prices depend on condition, "
    "location, service history and market demand."
)


# ---------- OVERVIEW METRICS ----------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Dataset records", f"{len(df):,}")
m2.metric("Car models", f"{df['name'].nunique():,}")
m3.metric("Average price", f"₹{df['selling_price'].mean():,.0f}")
m4.metric("Average kilometres", f"{df['km_driven'].mean():,.0f} km")

st.divider()


# ---------- NAVIGATION ----------
tab_predict, tab_compare, tab_analytics, tab_history, tab_about = st.tabs([
    "💰 Predict Price",
    "⚖️ Compare Cars",
    "📊 Analytics",
    "🧾 History",
    "ℹ️ About Project"
])


# ---------- PREDICT TAB ----------
with tab_predict:
    st.header("💰 Estimate Your Car's Price")
    st.write("Enter the vehicle details below to get a machine-learning estimate.")

    with st.form("prediction_form"):
        car = car_form("main", "Car details")
        submitted = st.form_submit_button(
            "✨ Predict Car Price",
            use_container_width=True
        )

    if submitted:
        try:
            price, model_input = predict_car(
                car["name"],
                car["year"],
                car["km_driven"],
                car["fuel"],
                car["transmission"],
                car["seller_type"],
                car["owner"]
            )

            st.session_state.last_prediction = price
            st.session_state.last_car = car
            st.session_state.last_model_input = model_input
            add_history(car, price)

        except Exception as e:
            st.error("Prediction nahi ho paayi. Error details:")
            st.code(str(e))

    if st.session_state.last_prediction is not None:
        price = st.session_state.last_prediction
        last_car = st.session_state.last_car

        st.divider()
        st.markdown("### Estimated Resale Price")
        st.markdown(
            f"<div style='background:#102a26;border:1px solid #10b981;"
            f"border-radius:18px;padding:24px;'>"
            f"<div style='color:#a7f3d0;font-size:16px;'>Predicted car value</div>"
            f"<div style='color:#34d399;font-size:42px;font-weight:800;'>"
            f"₹ {price:,.0f}</div>"
            f"<div style='color:#cbd5e1;'>{last_car['name']} • "
            f"{last_car['year']} • {last_car['km_driven']:,} km</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("Vehicle age", f"{max(0, current_year - last_car['year'])} years")
        r2.metric("Kilometres driven", f"{last_car['km_driven']:,} km")
        r3.metric("Fuel type", last_car["fuel"])

        st.caption(
            "This is a model-generated estimate, not a guaranteed selling price."
        )


# ---------- COMPARE TAB ----------
with tab_compare:
    st.header("⚖️ Compare Two Cars")
    st.write("Enter details for both cars to compare their estimated resale values.")

    with st.form("compare_form"):
        left, right = st.columns(2)

        with left:
            car_a = car_form("compare_a", "🚗 Car A")

        with right:
            car_b = car_form("compare_b", "🚙 Car B")

        compare_submitted = st.form_submit_button(
            "⚖️ Compare Prices",
            use_container_width=True
        )

    if compare_submitted:
        try:
            price_a, _ = predict_car(
                car_a["name"], car_a["year"], car_a["km_driven"],
                car_a["fuel"], car_a["transmission"],
                car_a["seller_type"], car_a["owner"]
            )
            price_b, _ = predict_car(
                car_b["name"], car_b["year"], car_b["km_driven"],
                car_b["fuel"], car_b["transmission"],
                car_b["seller_type"], car_b["owner"]
            )

            st.session_state.compare_result = {
                "car_a": car_a,
                "price_a": price_a,
                "car_b": car_b,
                "price_b": price_b
            }

            add_history(car_a, price_a)
            add_history(car_b, price_b)

        except Exception as e:
            st.error("Comparison nahi ho paaya.")
            st.code(str(e))

    if "compare_result" in st.session_state:
        result = st.session_state.compare_result
        a, b = st.columns(2)

        with a:
            st.markdown("### 🚗 Car A")
            st.subheader(result["car_a"]["name"])
            st.metric("Estimated price", f"₹{result['price_a']:,.0f}")
            st.write(f"Year: {result['car_a']['year']}")
            st.write(f"Kilometres: {result['car_a']['km_driven']:,} km")
            st.write(f"Fuel: {result['car_a']['fuel']}")

        with b:
            st.markdown("### 🚙 Car B")
            st.subheader(result["car_b"]["name"])
            st.metric("Estimated price", f"₹{result['price_b']:,.0f}")
            st.write(f"Year: {result['car_b']['year']}")
            st.write(f"Kilometres: {result['car_b']['km_driven']:,} km")
            st.write(f"Fuel: {result['car_b']['fuel']}")

        difference = abs(result["price_a"] - result["price_b"])
        st.divider()
        st.metric("Difference in estimated prices", f"₹{difference:,.0f}")

        if result["price_a"] > result["price_b"]:
            st.success(f"Car A has the higher estimated price by ₹{difference:,.0f}.")
        elif result["price_b"] > result["price_a"]:
            st.success(f"Car B has the higher estimated price by ₹{difference:,.0f}.")
        else:
            st.info("Both cars have the same estimated price.")


# ---------- ANALYTICS TAB ----------
with tab_analytics:
    st.header("📊 Dataset Analytics")
    st.write("Explore the prices in your dataset and, after predicting a car, see how its estimate changes.")

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("#### Average price by fuel type")
        fuel_chart = (
            df.groupby("fuel", as_index=False)["selling_price"]
            .mean()
            .sort_values("selling_price", ascending=False)
            .set_index("fuel")
        )
        st.bar_chart(fuel_chart)

    with chart_right:
        st.markdown("#### Average price by transmission")
        transmission_chart = (
            df.groupby("transmission", as_index=False)["selling_price"]
            .mean()
            .sort_values("selling_price", ascending=False)
            .set_index("transmission")
        )
        st.bar_chart(transmission_chart)

    st.markdown("#### Dataset price distribution")
    price_distribution = df[["selling_price"]].rename(
        columns={"selling_price": "Selling price (₹)"}
    )
    st.line_chart(
        price_distribution.reset_index(drop=True).head(100)
    )
    st.caption("The distribution chart shows the first 100 dataset records, not a time series.")

    if st.session_state.last_prediction is not None:
        last_car = st.session_state.last_car
        st.divider()
        st.markdown("### How this car's estimate changes")
        st.caption(
            f"Reference car: {last_car['name']}. Other details remain fixed "
            "while one input changes at a time."
        )

        age_rows = []
        max_age = max(1, current_year - min_year)
        for age in range(0, max_age + 1):
            test_year = current_year - age
            try:
                p, _ = predict_car(
                    last_car["name"], test_year, last_car["km_driven"],
                    last_car["fuel"], last_car["transmission"],
                    last_car["seller_type"], last_car["owner"]
                )
                age_rows.append({
                    "Vehicle age (years)": age,
                    "Estimated price (₹)": p
                })
            except Exception:
                continue

        km_points = sorted(set([
            0, 10000, 20000, 30000, 50000, 75000,
            100000, 150000, 200000, int(last_car["km_driven"])
        ]))
        km_rows = []

        for km in km_points:
            try:
                p, _ = predict_car(
                    last_car["name"], last_car["year"], km,
                    last_car["fuel"], last_car["transmission"],
                    last_car["seller_type"], last_car["owner"]
                )
                km_rows.append({
                    "Kilometres driven": km,
                    "Estimated price (₹)": p
                })
            except Exception:
                continue

        age_col, km_col = st.columns(2)

        with age_col:
            st.markdown("**Price vs vehicle age**")
            if age_rows:
                age_df = pd.DataFrame(age_rows).set_index("Vehicle age (years)")
                st.line_chart(age_df)

        with km_col:
            st.markdown("**Price vs kilometres driven**")
            if km_rows:
                km_df = pd.DataFrame(km_rows).set_index("Kilometres driven")
                st.line_chart(km_df)
    else:
        st.info("Predict a car in the Predict Price tab to unlock car-specific analytics.")

    st.divider()
    st.markdown("#### Dataset summary")
    st.dataframe(
        df[["year", "selling_price", "km_driven"]].describe().round(2),
        use_container_width=True
    )


# ---------- HISTORY TAB ----------

with tab_history:
    st.header("🧾 Prediction History")
    st.caption("History is kept for this session.")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_data,
            file_name="autovalue_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Your predictions will appear here after your first prediction.")

# ---------- ABOUT TAB ----------
with tab_about:
    st.header("ℹ️ About AutoValue")
    st.write(
        "AutoValue is a beginner-friendly machine learning project that estimates "
        "used-car resale prices from vehicle details."
    )

    st.markdown("#### Technologies used")
    st.markdown("""
    - **Python** — main programming language
    - **Pandas** — dataset loading and analysis
    - **Scikit-learn** — machine learning pipeline
    - **Tuned Extra Trees** — trained prediction model
    - **Streamlit** — interactive web dashboard
    """)

    st.markdown("#### Model inputs")
    st.markdown("""
    - Brand
    - Vehicle age
    - Kilometres driven
    - Fuel type
    - Transmission
    - Seller type
    - Previous owner
    """)

    st.markdown("#### Dataset")
    st.write(
        f"The local dataset contains {len(df):,} valid records and "
        f"{df['name'].nunique():,} distinct car model names."
    )

    st.warning(
        "Limitations: predictions depend on the training data and model quality. "
        "The app does not inspect a car's physical condition, accident history, "
        "service records or current local demand. Use estimates as guidance only."
    )


# ---------- FOOTER ----------
st.divider()
st.caption("AutoValue • Python • Pandas • Scikit-learn • Streamlit")
