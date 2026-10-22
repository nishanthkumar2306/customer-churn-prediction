"""Streamlit dashboard.   Run:  streamlit run dashboard/app.py"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.services import prediction_service as service
from src.data_preprocessing import ID_COL, TARGET, load_raw_data
from src.predict import LOW_MAX, MEDIUM_MAX, load_model, predict_dataframe, predict_single, results_to_csv_frame
from src.recommendations import DISCLAIMER
from src.train_model import METRICS_PATH

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")

RISK_COLORS = {"LOW": "#2e9e5b", "MEDIUM": "#e0a100", "HIGH": "#d64545"}
SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "sample_batch.csv"


# ---------------------------------------------------------------- data helpers
@st.cache_data(show_spinner="Scoring customers with the model...")
def load_scored_data() -> pd.DataFrame:
    """Raw dataset + model predictions for every customer."""
    raw = load_raw_data()
    scored, _ = predict_dataframe(raw, explain=False)
    scored.insert(0, ID_COL, raw[ID_COL].values)
    scored["Churned"] = (raw[TARGET] == "Yes").values
    return scored.drop(columns=["risk_factors", "protective_factors", "retention_suggestions"])


def risk_badge(level: str) -> str:
    return (f"<span style='background:{RISK_COLORS[level]};color:white;padding:4px 14px;"
            f"border-radius:12px;font-weight:700'>{level} RISK</span>")


def rate_chart(df, col, title, order=None):
    g = df.groupby(col, observed=True)["Churned"].agg(["mean", "count"]).reset_index()
    fig = px.bar(g, x=col, y="mean", text=g["mean"].map("{:.0%}".format), title=title,
                 category_orders={col: order} if order else None,
                 labels={"mean": "Churn rate", col: ""}, hover_data={"count": True})
    fig.update_yaxes(tickformat=".0%", range=[0, max(0.6, g["mean"].max() * 1.15)])
    fig.update_layout(height=320, margin=dict(t=50, b=10))
    return fig


def show_result(result: dict):
    """Shared by 'Predict Customer' and 'Customer Analysis'."""
    p, level = result["churn_probability"], result["risk_level"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", result["prediction"])
    c2.metric("Churn probability", f"{p:.0%}")
    c3.markdown("**Risk level**")
    c3.markdown(risk_badge(level), unsafe_allow_html=True)
    st.progress(min(max(p, 0.0), 1.0))

    left, right = st.columns(2)
    with left:
        st.subheader("Important factors influencing the model prediction")
        if result["important_factors"]:
            for f in result["important_factors"]:
                st.markdown(f"- **{f['feature']}** = {f['value']} (raises predicted probability "
                            f"by about {f['effect_on_probability'] * 100:.0f} points)")
        else:
            st.write("No factor raised the predicted probability noticeably.")
        if result["protective_factors"]:
            st.caption("Factors lowering the prediction: " + ", ".join(
                f"{f['feature']} = {f['value']}" for f in result["protective_factors"]))
        st.caption("These describe how the model reacts to this customer's data. "
                   "They do not prove what causes churn.")
    with right:
        st.subheader("Retention suggestions")
        for s in result["retention_suggestions"]:
            st.markdown(f"- **{s['action']}**  \n  _{s['reason']}_")
        st.caption(DISCLAIMER)


# ----------------------------------------------------------------------- pages
def page_dashboard():
    st.title("Churn Dashboard")
    df = load_scored_data()
    k = st.columns(5)
    k[0].metric("Total customers", f"{len(df):,}")
    k[1].metric("Churned customers", f"{int(df['Churned'].sum()):,}")
    k[2].metric("Churn rate", f"{df['Churned'].mean():.1%}")
    k[3].metric("Avg monthly charges", f"${df['MonthlyCharges'].mean():.2f}")
    k[4].metric("High-risk customers", f"{int((df['risk_level'] == 'HIGH').sum()):,}")
    st.caption("High-risk counts come from the trained model applied to this same dataset "
               "(which it was trained on), so treat them as a demonstration.")

    a, b = st.columns(2)
    dist = df["Churned"].map({True: "Churned", False: "Stayed"}).value_counts().reset_index()
    dist.columns = ["Status", "Customers"]
    a.plotly_chart(px.pie(dist, names="Status", values="Customers", title="Churn distribution",
                          hole=0.45, color="Status",
                          color_discrete_map={"Churned": "#d64545", "Stayed": "#2e9e5b"}
                          ).update_layout(height=320, margin=dict(t=50, b=10)), width="stretch")
    risk = df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).reset_index()
    risk.columns = ["Risk", "Customers"]
    b.plotly_chart(px.bar(risk, x="Risk", y="Customers", color="Risk", color_discrete_map=RISK_COLORS,
                          title="Predicted risk distribution").update_layout(
        height=320, margin=dict(t=50, b=10), showlegend=False), width="stretch")

    df["Tenure group"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                                labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"])
    df["Monthly charge group"] = pd.cut(df["MonthlyCharges"], [0, 35, 70, 90, 200],
                                        labels=["<$35", "$35-70", "$70-90", ">$90"])
    df["Senior"] = df["SeniorCitizen"].map({"Yes": "Senior", "No": "Not senior"})
    charts = [
        ("Contract", "Churn rate by contract type", None), ("Tenure group", "Churn rate by tenure", None),
        ("Monthly charge group", "Churn rate by monthly charges", None),
        ("PaymentMethod", "Churn rate by payment method", None),
        ("InternetService", "Churn rate by internet service", None),
        ("Senior", "Churn rate by senior citizen status", None),
        ("TechSupport", "Churn rate by tech support", None),
        ("OnlineSecurity", "Churn rate by online security", None),
    ]
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for col_ui, (col, title, order) in zip(cols, charts[i:i + 2]):
            col_ui.plotly_chart(rate_chart(df, col, title, order), width="stretch")


def page_predict():
    st.title("Predict Customer Churn")
    st.write("Fill in the customer details. Only tenure, monthly charges and contract are required in the API; "
             "here every field is available.")
    yn = ["No", "Yes"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Profile**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", yn)
        partner = st.selectbox("Partner", yn)
        dependents = st.selectbox("Dependents", yn)
        tenure = st.number_input("Tenure (months)", 0, 120, 5)
    with c2:
        st.markdown("**Services**")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        lines = st.selectbox("Multiple lines", ["No phone service"] if phone == "No" else yn)
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        addon = ["No internet service"] if internet == "No" else yn
        security = st.selectbox("Online security", addon)
        backup = st.selectbox("Online backup", addon)
        protection = st.selectbox("Device protection", addon)
    with c3:
        st.markdown("**Support, streaming & billing**")
        support = st.selectbox("Tech support", addon)
        tv = st.selectbox("Streaming TV", addon)
        movies = st.selectbox("Streaming movies", addon)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges ($)", 0.0, 1000.0, 85.5, step=0.5)
    total = st.number_input("Total charges ($)  (0 = estimate from tenure x monthly)", 0.0, 200000.0, 0.0)

    if st.button("Predict churn", type="primary"):
        customer = {"gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
                    "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
                    "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
                    "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
                    "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
                    "PaymentMethod": payment, "MonthlyCharges": monthly,
                    "TotalCharges": total if total > 0 else None}
        st.divider()
        show_result(predict_single(customer))


def page_batch():
    st.title("Batch Prediction (CSV upload)")
    st.write("Upload a CSV with Telco-style columns. Required: `tenure`, `MonthlyCharges`, `Contract`. "
             f"Limits: {service.MAX_UPLOAD_BYTES // (1024 * 1024)} MB and {service.MAX_ROWS} rows.")
    if SAMPLE_CSV.exists():
        st.download_button("Download a sample CSV", SAMPLE_CSV.read_bytes(), "sample_batch.csv", "text/csv")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is None:
        return
    try:
        df = service.read_csv_bytes(uploaded.getvalue())
        with st.spinner("Predicting..."):
            flat, warnings = service.predict_batch(df)
    except service.InvalidUpload as exc:
        st.error(f"Could not process this file: {exc}")
        return
    except Exception:
        st.error("Something went wrong while processing the file. Please check its format.")
        return
    for w in warnings:
        st.warning(w)
    st.success(f"Predicted {len(flat)} customers.")
    counts = flat["risk_level"].value_counts()
    cols = st.columns(3)
    for col_ui, level in zip(cols, ["LOW", "MEDIUM", "HIGH"]):
        col_ui.metric(f"{level} risk", int(counts.get(level, 0)))
    show_cols = [c for c in ["customerID", "churn_probability", "prediction", "risk_level", "Contract",
                             "MonthlyCharges", "tenure", "important_factors", "retention_suggestions"]
                 if c in flat.columns]
    st.dataframe(flat[show_cols].sort_values("churn_probability", ascending=False),
                 width="stretch", hide_index=True,
                 column_config={"churn_probability": st.column_config.ProgressColumn(
                     "Churn probability", min_value=0.0, max_value=1.0, format="percent")})
    st.download_button("Download results as CSV", flat.to_csv(index=False).encode("utf-8"),
                       "churn_predictions.csv", "text/csv")
    st.caption(DISCLAIMER + " Risk factors are listed only for MEDIUM and HIGH risk customers.")


def page_risk():
    st.title("Risk Analysis & High-Risk Customers")
    st.write(f"Risk levels: **LOW** below {LOW_MAX:.0%}, **MEDIUM** {LOW_MAX:.0%}–{MEDIUM_MAX:.0%}, "
             f"**HIGH** {MEDIUM_MAX:.0%} and above.")
    df = load_scored_data()
    f1, f2, f3 = st.columns(3)
    levels = f1.multiselect("Risk level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH"])
    contracts = f2.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
    min_prob = f3.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05)
    view = df[df["risk_level"].isin(levels) & df["Contract"].isin(contracts)
              & (df["churn_probability"] >= min_prob)]
    view = view.sort_values("churn_probability", ascending=False)
    st.write(f"**{len(view):,}** customers match the filters (click a column header to sort).")
    table = view[[ID_COL, "churn_probability", "risk_level", "Contract", "MonthlyCharges", "tenure"]]
    st.dataframe(table, width="stretch", hide_index=True, column_config={
        "churn_probability": st.column_config.ProgressColumn("Churn probability", min_value=0.0,
                                                             max_value=1.0, format="percent")})
    st.download_button("Download this list as CSV", table.to_csv(index=False).encode("utf-8"),
                       "high_risk_customers.csv", "text/csv")
    by_contract = df.groupby(["Contract", "risk_level"]).size().reset_index(name="Customers")
    st.plotly_chart(px.bar(by_contract, x="Contract", y="Customers", color="risk_level",
                           color_discrete_map=RISK_COLORS, barmode="stack",
                           title="Predicted risk level by contract type").update_layout(height=340),
                    width="stretch")


def page_customer():
    st.title("Customer Analysis")
    raw = load_raw_data()
    customer_id = st.selectbox("Select a customer ID", raw[ID_COL].tolist())
    row = raw[raw[ID_COL] == customer_id].iloc[[0]]
    st.subheader("Customer information")
    info = row.drop(columns=[ID_COL, TARGET]).iloc[0]
    cols = st.columns(4)
    for i, (name, value) in enumerate(info.items()):
        cols[i % 4].markdown(f"**{name}**  \n{value}")
    st.caption(f"Customer ID: {customer_id}  |  Recorded outcome in dataset: "
               f"{'Churned' if row[TARGET].iloc[0] == 'Yes' else 'Stayed'}")
    st.divider()
    results, _ = predict_dataframe(row)
    r = results.iloc[0]
    show_result({"prediction": r["prediction"], "churn_probability": float(r["churn_probability"]),
                 "risk_level": r["risk_level"], "important_factors": r["risk_factors"],
                 "protective_factors": r["protective_factors"],
                 "retention_suggestions": r["retention_suggestions"]})


def page_model():
    st.title("Model Information")
    bundle = load_model()
    report = json.loads(METRICS_PATH.read_text())
    st.write(f"**Dataset:** {report['dataset']} ({report['n_rows']:,} customers, "
             f"{report['churn_rate']:.1%} churned)  \n**Selected model:** {bundle['model_name']} "
             "(chosen by cross-validated ROC-AUC on the training data)  \n"
             f"**Test split:** {report['test_size']:.0%}, random_state={report['random_state']}  \n"
             f"**Decision threshold:** {report['decision_threshold']} (probability ≥ this = 'Likely to Churn')")
    rows = []
    for name, m in report["models"].items():
        rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"],
                     "CV ROC-AUC": m["cv_roc_auc_mean"]})
    st.subheader("Test-set results")
    st.dataframe(pd.DataFrame(rows).set_index("Model").round(3), width="stretch")

    cm = report["models"][bundle["model_name"]]["confusion_matrix"]
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = px.imshow(z, text_auto=True, x=["Predicted stay", "Predicted churn"],
                    y=["Actually stayed", "Actually churned"], color_continuous_scale="Blues",
                    title=f"Confusion matrix – {bundle['model_name']}")
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Limitations")
    st.markdown("- Predictions are probabilities, not certainties.\n"
                "- Historical data may contain bias; the model learns patterns from one telecom dataset.\n"
                "- Correlation does not imply causation: influencing factors are not proven causes.\n"
                "- Recommendations are rule-based suggestions, not guaranteed fixes.\n"
                "- Model quality depends on dataset quality.\n"
                "- Real retention decisions need business context.")


PAGES = {"Dashboard": page_dashboard, "Predict Customer": page_predict,
         "Batch Prediction": page_batch, "Risk Analysis": page_risk,
         "Customer Analysis": page_customer, "Model Information": page_model}

st.sidebar.title("📉 Churn Prediction")
choice = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.caption("Customer Churn Prediction, Risk Analysis & Retention Recommendation System")
try:
    PAGES[choice]()
except FileNotFoundError as exc:
    st.error(f"{exc}")
