import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection")
st.subheader("AI Powered Real-time Fraud Detection")
st.markdown("---")

@st.cache_data
def load_examples():
    df = pd.read_csv('creditcard.csv')
    fraud = df[df['Class']==1].head(20)
    safe = df[df['Class']==0].head(20)
    return fraud, safe

fraud_df, safe_df = load_examples()

mode = st.radio(
    "Select Demo Mode:",
    ["🟢 Safe Transaction", "🔴 Fraudulent Transaction"]
)

if mode == "🟢 Safe Transaction":
    example = safe_df.sample(1).iloc[0]
    st.info("Loading a SAFE transaction from real data")
else:
    example = fraud_df.sample(1).iloc[0]
    st.warning("Loading a FRAUDULENT transaction from real data")

col1, col2 = st.columns(2)
with col1:
    st.metric("💰 Amount", f"₹{example['Amount']:.2f}")
with col2:
    st.metric("⏰ Time", f"{example['Time']:.0f}")

st.markdown("---")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_btn = st.button("🔍 Detect Fraud", use_container_width=True)

if predict_btn:
    # FastAPI ko call karo
    features = example.drop('Class').tolist()
    
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"features": features}
    )
    
    result = response.json()
    
    st.markdown("---")
    st.markdown("### 🎯 AI Result")
    
    if result['prediction'] == 1:
        st.error(f"""
        ## ⚠️ FRAUD DETECTED!
        **Fraud Probability: {result['fraud_probability']}%**
        🚨 Block this transaction immediately!
        """)
    else:
        st.success(f"""
        ## ✅ TRANSACTION IS SAFE!
        **Safe Probability: {result['safe_probability']}%**
        ✅ Transaction appears legitimate!
        """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Amount", f"₹{example['Amount']:.2f}")
    col2.metric("Actual", "FRAUD" if example['Class']==1 else "SAFE")
    col3.metric("AI Said", result['result'])

st.markdown("---")
st.markdown("Built with ❤️ by **Khushal** | Fraud Detection AI")