"""
Streamlit Frontend — Laptop Price Prediction
Starten: streamlit run app/streamlit_app.py
"""

import base64
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Laptop-Preisvorhersage",
    page_icon="💻",
    layout="wide",
)

# ─── Style ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .price-box {
        background: #1E293B; border-radius: 12px; padding: 1.5rem;
        text-align: center; margin-bottom: 1rem;
    }
    .price-value { font-size: 3rem; font-weight: 700; color: #3B82F6; }
    .price-range { font-size: 1rem; color: #94A3B8; margin-top: 0.3rem; }
    .section-title { font-size: 1.1rem; font-weight: 600; color: #E2E8F0;
                     margin: 1.2rem 0 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar — Input Form ─────────────────────────────────────────────────────

with st.sidebar:
    st.title("💻 Laptop-Specs")
    st.markdown("---")

    brand = st.selectbox("Marke", [
        "Lenovo", "HP", "Dell", "Asus", "Acer", "Apple",
        "Microsoft", "Samsung", "MSI", "LG", "Andere",
    ])

    cpu = st.selectbox("Prozessor", [
        "Intel Core i3", "Intel Core i5", "Intel Core i7", "Intel Core i9",
        "Intel Core Ultra 5", "Intel Core Ultra 7", "Intel Core Ultra 9",
        "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7", "AMD Ryzen 9",
        "Apple M1", "Apple M2", "Apple M3", "Apple M3 Pro", "Apple M3 Max",
        "Snapdragon X Elite",
    ])

    gpu = st.selectbox("Grafikkarte", [
        "Integriert (Intel UHD / AMD Radeon)",
        "NVIDIA RTX 4050", "NVIDIA RTX 4060", "NVIDIA RTX 4070", "NVIDIA RTX 4080",
        "NVIDIA RTX 3060", "NVIDIA RTX 3070",
        "AMD RX 6600M", "AMD RX 7600M",
        "Apple GPU (integriert)",
    ])

    ram_gb = st.select_slider("RAM (GB)", options=[4, 8, 16, 32, 64], value=16)

    storage_gb = st.select_slider("Speicher (GB)", options=[128, 256, 512, 1024, 2048], value=512)
    storage_type = st.radio("Speichertyp", ["SSD", "HDD"], horizontal=True)

    display_inch = st.slider("Display-Größe (Zoll)", 11.0, 18.0, 15.6, step=0.1)

    resolution = st.selectbox("Auflösung", [
        "1920x1080 (Full HD)", "2560x1600 (QHD)", "3840x2160 (4K UHD)",
        "2560x1440 (QHD)", "1366x768 (HD)", "2880x1800 (Retina)",
    ])

    os_choice = st.selectbox("Betriebssystem", [
        "Windows 11", "Windows 11 Home", "Windows 11 Pro",
        "macOS", "Linux", "Ohne OS",
    ])

    weight_kg = st.slider("Gewicht (kg)", 0.9, 4.0, 1.8, step=0.1)

    st.markdown("---")
    predict_btn = st.button("Preis vorhersagen", type="primary", use_container_width=True)

# ─── Main Area ────────────────────────────────────────────────────────────────

st.title("Laptop-Preisvorhersage")
st.markdown("*End-to-End ML Pipeline · MLflow · SHAP Explainability*")

tab_predict, tab_shap = st.tabs(["Vorhersage", "SHAP Analyse"])

with tab_predict:
    if predict_btn:
        payload = {
            "brand":        brand,
            "cpu":          cpu,
            "gpu":          gpu,
            "ram_gb":       ram_gb,
            "storage_gb":   storage_gb,
            "storage_type": storage_type,
            "display_inch": display_inch,
            "resolution":   resolution.split(" ")[0],
            "os":           os_choice,
            "weight_kg":    weight_kg,
        }

        with st.spinner("Berechne Preis …"):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("API nicht erreichbar. Starte zuerst: `uvicorn api.main:app --reload`")
                st.stop()
            except Exception as e:
                st.error(f"Fehler: {e}")
                st.stop()

        col1, col2 = st.columns([1, 2])

        with col1:
            price = data["predicted_price"]
            low, high = data["confidence_interval"]
            st.markdown(f"""
            <div class="price-box">
                <div class="price-value">€ {price:,.0f}</div>
                <div class="price-range">Konfidenzintervall: €{low:,.0f} – €{high:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-title">Top SHAP-Treiber</div>', unsafe_allow_html=True)
            shap_vals = data["shap_values"]
            shap_sorted = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            for feat, val in shap_sorted:
                color = "#10B981" if val > 0 else "#F87171"
                sign  = "+" if val > 0 else ""
                st.markdown(
                    f"<span style='color:{color};font-weight:600'>{sign}€{val:,.1f}</span>"
                    f"  `{feat}`",
                    unsafe_allow_html=True,
                )

        with col2:
            if data.get("shap_plot_base64"):
                st.markdown('<div class="section-title">SHAP Waterfall-Chart</div>',
                            unsafe_allow_html=True)
                img_bytes = base64.b64decode(data["shap_plot_base64"])
                st.image(img_bytes, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Modell-Vergleich (alle MLflow-Runs)</div>',
                    unsafe_allow_html=True)
        try:
            runs_resp = requests.get(f"{API_URL}/models", timeout=10)
            if runs_resp.ok:
                runs = runs_resp.json()
                if runs:
                    df_runs = pd.DataFrame(runs)
                    metrics_df = pd.json_normalize(df_runs["metrics"])
                    display_cols = [c for c in ["rmse", "mae", "r2", "mape"] if c in metrics_df.columns]
                    df_show = pd.concat([df_runs[["run_name"]], metrics_df[display_cols]], axis=1)
                    df_show.columns = ["Modell"] + [c.upper() for c in display_cols]
                    st.dataframe(df_show.style.highlight_max(subset=["R2"], color="#0F3460")
                                 .highlight_min(subset=["RMSE", "MAE"], color="#0F3460"),
                                 use_container_width=True)
        except Exception:
            st.info("Modell-Vergleich nicht verfügbar (MLflow-Server läuft?)")

    else:
        st.info("Laptop-Specs in der Sidebar einstellen und auf **Preis vorhersagen** klicken.")

        with st.expander("Wie funktioniert das?"):
            st.markdown("""
            1. **Scraper** — Täglich werden Laptop-Daten von MediaMarkt gescrapt
            2. **Feature Engineering** — CPU-Score, GPU-Tier, RAM, Speicher, Display-Features
            3. **MLflow** — 4 Modelle (Linear, RandomForest, XGBoost, LightGBM) werden verglichen
            4. **SHAP** — Erklärt, welche Spec wie viel zum Preis beiträgt
            5. **FastAPI** — REST-API liefert Prognose + SHAP-Plot
            6. **Diese App** — Streamlit-Frontend für interaktive Preisprognose
            """)

with tab_shap:
    st.markdown("### SHAP Global Feature Importance")
    st.markdown(
        "Zeigt, welche Laptop-Merkmale den Preis über alle Trainingsdaten hinweg "
        "am stärksten beeinflussen."
    )
    try:
        shap_resp = requests.get(f"{API_URL}/shap/summary", timeout=15)
        if shap_resp.ok:
            shap_data = shap_resp.json()
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Beeswarm Summary Plot**")
                st.caption("Jeder Punkt = ein Laptop. Farbe = Feature-Wert (rot=hoch, blau=niedrig).")
                st.image(base64.b64decode(shap_data["summary_b64"]), use_container_width=True)
            with col_b:
                if shap_data.get("bar_b64"):
                    st.markdown("**Mean |SHAP| — Globale Wichtigkeit**")
                    st.caption("Durchschnittlicher absoluter SHAP-Wert je Feature.")
                    st.image(base64.b64decode(shap_data["bar_b64"]), use_container_width=True)
        elif shap_resp.status_code == 404:
            st.warning("SHAP-Plots noch nicht generiert. Training einmal ausführen:\n"
                       "```\npython -m src.train --data data/raw/laptops_synthetic_2026-06-15.csv\n```")
        else:
            st.error(f"API-Fehler: {shap_resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("API nicht erreichbar. Starte zuerst: `uvicorn api.main:app --reload`")
