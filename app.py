import time, json
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)

st.set_page_config(page_title="Fraud on LiveStream", layout="centered")

#  Modelo + threshold
@st.cache_resource
def load_model_and_threshold():
    model = CatBoostClassifier()
    model.load_model("fraude_model.cbm")
    try:
        with open("fraude_threshold.json") as f:
            thr = float(json.load(f)["threshold"])
    except Exception:
        thr = 0.5
    return model, thr

model, saved_thr = load_model_and_threshold()

# ordered columns
EXPECTED = [c for c in [
    "Time", *[f"V{i}" for i in range(1,29)], "Amount" 
] if c]

st.title("Fraud Detection — LiveStream Monitor")

src = st.radio("Data Source", ["Upload csv", "Use demo (sample_creditcard_demo.csv)"], horizontal=True)

uploaded = None
if src == "Upload csv":
    uploaded = st.file_uploader("Upload a csv with same train columns", type=["csv"])

thr = st.sidebar.slider("Threshold", 0.01, 0.99, float(saved_thr), 0.01)
autoplay = st.sidebar.checkbox("Auto-play", value=False)
delay = st.sidebar.slider("Velocity (seg)", 0.1, 3.0, 0.8, 0.1)
topk_mode = st.sidebar.checkbox("Top-K (ignore threshold and mark most dangerous K's)")
K = st.sidebar.number_input("K casos", min_value=1, value=100, step=10)

if "idx" not in st.session_state:
    st.session_state.idx = 0
if "running" not in st.session_state:
    st.session_state.running = False

# Load dataframe
if src == "Use demo (sample_creditcard_demo.csv)":
    df = pd.read_csv("sample_creditcard_demo.csv")
else:
    if uploaded is None:
        st.info("Upload a csv or change to demo source.")
        st.stop()
    df = pd.read_csv(uploaded)

has_label = "Class" in df.columns
X = df.drop(columns=["Class"]) if has_label else df.copy()

# Adjust columns to spected order if all of these exist
if set(EXPECTED).issubset(set(X.columns)):
    X = X[EXPECTED]  # reorder

n = len(X)
if n == 0:
    st.error("There are no rows to show.")
    st.stop()

# COntrols
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("◀️ Reboot"):
        st.session_state.idx = 0; st.session_state.running = False
with c2:
    if st.button("⏭️ Next"):
        st.session_state.idx = min(st.session_state.idx + 1, n - 1)
with c3:
    if st.button("▶️ Auto" if not st.session_state.running else "⏸️ Pause"):
        st.session_state.running = not st.session_state.running

st.write("---")

# render 1
i = st.session_state.idx
row = X.iloc[i:i+1]
probs = model.predict_proba(row)[:, 1]
if topk_mode:
    # mark top-k on all currently batch
    all_probs = model.predict_proba(X)[:, 1]
    k = min(int(K), len(all_probs))
    thr_topk = np.partition(all_probs, -k)[-k]
    pred = int(float(probs[0]) >= thr_topk)
    thr_label = f"(Top-K, thr≈{thr_topk:.3f})"
else:
    pred = int(float(probs[0]) >= float(thr))
    thr_label = f"(thr={float(thr):.3f})"

pred_txt = "⚠️ FRAUD" if pred==1 else "✅ Normal"
st.subheader(f"Transaction #{i+1}/{n}  {thr_label}")
st.metric("Prediction", pred_txt, None)
st.markdown(f"Fraud probability: **{float(probs[0]):.3f}**")
st.progress((i+1)/n)

st.caption("Transaction features")
st.dataframe(row.T.rename(columns={i: "value"}))

if has_label:
    real = int(df["Class"].iloc[i])
    ok = "✔" if real==pred else "✖"
    st.write(f"Real: **{real}**  |  Pred: **{pred}**  {ok}")

# Auto-play
if autoplay or st.session_state.running:
    time.sleep(float(delay))
    st.session_state.idx = (st.session_state.idx + 1) % n
    st.rerun()

# Resume/Annexes (opcional if there is Class)
with st.expander("Batch Resume (If csv includes 'Class')"):
    if has_label:
        all_probs = model.predict_proba(X)[:, 1]
        if topk_mode:
            y_pred_all = (all_probs >= thr_topk).astype(int)
        else:
            y_pred_all = (all_probs >= float(thr)).astype(int)
        y_true = df["Class"].values

        cm = confusion_matrix(y_true, y_pred_all)
        st.write(pd.DataFrame(cm,
                              index=["Real 0 (Normal)","Real 1 (Fraud)"],
                              columns=["Pred 0 (Normal)","Pred 1 (Fraud)"]))
        st.text(classification_report(y_true, y_pred_all, digits=4))

        # Fast curves
        prec, rec, _ = precision_recall_curve(y_true, all_probs)
        ap = average_precision_score(y_true, all_probs)
        fpr, tpr, _ = roc_curve(y_true, all_probs)
        roc_auc = auc(fpr, tpr)

        st.line_chart(pd.DataFrame({"Recall": rec, "Precision": prec}))
        st.caption(f"PR-AUC: {ap:.3f} | ROC-AUC: {roc_auc:.3f}")
    else:
        st.info("For global metrics, csv must to have 'Class' column.")
