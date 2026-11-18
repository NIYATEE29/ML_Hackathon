import json
import time
import requests
import streamlit as st
import streamlit.components.v1 as components

# Configure page layout
st.set_page_config(
    page_title="PathSnap Route Reconstruction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished hackathon presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">PathSnap: Map-Matching & Route Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">PES University ML Hackathon (UE24CS352A) &bull; Distributed Laptop 1 (FastAPI) + Laptop 2 (UI)</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("Backend Configuration")
api_url = st.sidebar.text_input("FastAPI Backend URL", value="http://localhost:8000")

# Check Backend Status
backend_online = False
try:
    health_resp = requests.get(f"{api_url}/", timeout=3)
    if health_resp.status_code == 200:
        backend_online = True
        st.sidebar.success("Backend Connected (Laptop 1 Online)")
    else:
        st.sidebar.error(f"Backend Returned {health_resp.status_code}")
except Exception:
    st.sidebar.error("Cannot connect to Backend API. Ensure Laptop 1 backend is running.")

split_choice = st.sidebar.selectbox("Data Partition", ["student_test", "train"], index=0)

# Fetch Available Trips
trips_list = []
if backend_online:
    try:
        trips_resp = requests.get(f"{api_url}/trips?split={split_choice}", timeout=5)
        if trips_resp.status_code == 200:
            data = trips_resp.json()
            trips_list = [t["trip_id"] for t in data.get("trips", [])]
    except Exception as e:
        st.sidebar.warning(f"Error fetching trips: {e}")

if not trips_list:
    # Default fallback trip IDs
    trips_list = ["00000040", "00000057", "00000065", "00000076", "00000001"]

selected_trip = st.sidebar.selectbox("Select Evaluation Trip", trips_list, index=0)
eval_mode = st.sidebar.checkbox("Server-Side Evaluation Mode", value=True, help="Evaluates accuracy against ground-truth route on the server side without leaking labels.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Team Work Split:**")
st.sidebar.markdown("- **Person A:** Logistic Regression + KNN")
st.sidebar.markdown("- **Person B:** Random Forest / Extra Trees")
st.sidebar.markdown("- **Person C:** LightGBM / XGBoost")
st.sidebar.markdown("- **Person D (Ours):** MLP + Features & HMM Pipeline")

# Main Interface
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.metric("Selected Trip ID", selected_trip)
with col_info2:
    st.metric("Partition", split_choice)
with col_info3:
    st.metric("Scoring Model", "MLP (64, 32)")
with col_info4:
    st.metric("Route Decoder", "HMM + Viterbi")

btn_run = st.button("🚀 Reconstruct Route & Match Map", type="primary", use_container_width=True)

if btn_run or f"results_{selected_trip}" in st.session_state:
    if btn_run:
        with st.spinner("Executing End-to-End Pipeline on Laptop 1 (Candidates -> MLP -> HMM Viterbi)..."):
            t_start = time.time()
            try:
                # 1. Prediction Call
                pred_resp = requests.post(
                    f"{api_url}/predict",
                    json={"trip_id": selected_trip, "split": split_choice, "use_cached_features": True},
                    timeout=60
                )
                if pred_resp.status_code != 200:
                    st.error(f"Prediction failed: {pred_resp.text}")
                    st.stop()
                pred_data = pred_resp.json()

                eval_data = {}
                if eval_mode:
                    # 2. Server-Side Evaluation Call
                    ev_resp = requests.post(
                        f"{api_url}/evaluate",
                        json={"trip_id": selected_trip, "split": split_choice, "use_cached_features": True},
                        timeout=60
                    )
                    if ev_resp.status_code == 200:
                        eval_data = ev_resp.json()

                # 3. Fetch Folium Map HTML
                map_resp = requests.get(f"{api_url}/map/{selected_trip}?split={split_choice}", timeout=30)
                map_html = map_resp.text if map_resp.status_code == 200 else None

                st.session_state[f"results_{selected_trip}"] = {
                    "pred": pred_data,
                    "eval": eval_data,
                    "map_html": map_html
                }
            except Exception as ex:
                st.error(f"Failed to communicate with Backend API: {ex}")
                st.stop()

    res = st.session_state[f"results_{selected_trip}"]
    pred = res["pred"]
    ev = res["eval"]
    map_html = res["map_html"]

    # Display Metrics Scorecard
    st.markdown("### 📊 Performance Scorecard")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        acc = ev.get("hmm_top1_accuracy")
        clf_acc = ev.get("classifier_top1_accuracy")
        if acc is not None:
            delta = f"+{(acc - clf_acc)*100:.1f}% vs Clf" if clf_acc is not None else None
            st.metric("HMM Top-1 Accuracy", f"{acc * 100:.2f}%", delta=delta)
        else:
            st.metric("Reconstructed Points", pred["num_points"])

    with m2:
        hmm_cont = pred.get("hmm_continuity", 0.0)
        clf_cont = pred.get("classifier_continuity", 0.0)
        st.metric("HMM Route Continuity", f"{hmm_cont * 100:.1f}%", delta=f"+{(hmm_cont - clf_cont)*100:.1f}% vs Clf")

    with m3:
        jaccard = ev.get("route_jaccard_similarity")
        if jaccard is not None:
            st.metric("Route Arc Jaccard Overlap", f"{jaccard * 100:.2f}%")
        else:
            st.metric("Classifier Continuity", f"{clf_cont * 100:.1f}%")

    with m4:
        tot_time = pred["timings"].get("total_pipeline_ms", 0.0)
        st.metric("End-to-End Latency", f"{tot_time:.1f} ms", f"{tot_time / max(1, pred['num_points']):.2f} ms / point")

    # Latency Breakdown Expandable
    with st.expander("⏱️ Detailed Latency & Pipeline Timing Breakdown"):
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("Candidate Retrieval", f"{pred['timings'].get('candidate_gen_ms', 0):.1f} ms")
        t_col2.metric("MLP Scoring", f"{pred['timings'].get('mlp_inference_ms', 0):.1f} ms")
        t_col3.metric("HMM Viterbi", f"{pred['timings'].get('hmm_viterbi_ms', 0):.1f} ms")
        t_col4.metric("Total Wall Time", f"{tot_time:.1f} ms")

    # Map Visualization
    st.markdown("### 🗺️ Interactive Folium Map Matching")
    if map_html:
        components.html(map_html, height=520, scrolling=True)
    else:
        st.info("Map preview unavailable.")

    # Reconstructed Sequence Inspection
    with st.expander("📜 View Chronological Reconstructed Arc Sequence"):
        route_arcs = pred["predicted_route"]
        st.write(f"Total predicted arc sequence length: **{len(route_arcs)}** arcs")
        st.code(str(route_arcs[:100]) + ("..." if len(route_arcs) > 100 else ""))
