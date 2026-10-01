import os
import glob
import json
import io
from datetime import datetime
import pandas as pd
import streamlit as st

# Import core multi-agent components
from src.models import call_llm, load_model_and_tokenizer
from src.tools import load_and_clean_data, perform_eda, generate_plot, calculate_correlation, generate_report
from src.orchestrator import orchestrator_agent
from src.config import DEVICE, MODEL_NAME, HF_TOKEN

# Page Configuration
st.set_page_config(
    page_title="Multi-Agent Data Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #6366F1, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-data { background-color: #1E3A8A; color: #93C5FD; }
    .badge-eda { background-color: #064E3B; color: #6EE7B7; }
    .badge-viz { background-color: #4C1D95; color: #C4B5FD; }
    .badge-analysis { background-color: #78350F; color: #FCD34D; }
    .badge-report { background-color: #831843; color: #F472B6; }
    .stat-card {
        border-radius: 8px;
        padding: 12px;
        background-color: #1E293B;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "eda_results" not in st.session_state:
    st.session_state.eda_results = {}
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "plot_history" not in st.session_state:
    st.session_state.plot_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "report" not in st.session_state:
    st.session_state.report = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Helper: Agent Badge HTML
def get_agent_badge(agent_name: str) -> str:
    agent_map = {
        "data": ("Data Ingestion Agent", "badge-data"),
        "eda": ("EDA Agent", "badge-eda"),
        "visualization": ("Visualization Agent", "badge-viz"),
        "analysis": ("Statistical Analysis Agent", "badge-analysis"),
        "report": ("Report Generation Agent", "badge-report"),
        "orchestrator": ("Central Orchestrator", "badge-viz")
    }
    label, cls = agent_map.get(agent_name.lower(), (agent_name.title(), "badge-data"))
    return f'<span class="agent-badge {cls}">🤖 {label}</span>'

# Sidebar: Data Source & Configuration
with st.sidebar:
    st.markdown("## 📁 Dataset Management")
    data_source_mode = st.radio(
        "Select Data Source",
        ["Upload CSV", "Sample: Tips Dataset", "Sample: Iris Dataset"],
        index=0
    )

    if data_source_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_file is not None and (st.session_state.dataset_name != uploaded_file.name):
            with st.spinner("Loading and cleaning dataset..."):
                result = load_and_clean_data(uploaded_file)
                if result["status"] == "success":
                    st.session_state.df = result["data"]
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.eda_results = {}
                    st.session_state.analysis_results = {}
                    st.session_state.plot_history = []
                    st.session_state.report = None
                    st.success(f"Loaded: {uploaded_file.name}")
                else:
                    st.error(result["message"])
    else:
        sample_path = "sample_data/tips.csv" if "Tips" in data_source_mode else "sample_data/iris.csv"
        sample_name = os.path.basename(sample_path)
        if st.session_state.dataset_name != sample_name:
            if os.path.exists(sample_path):
                with st.spinner(f"Loading {sample_name}..."):
                    result = load_and_clean_data(sample_path)
                    if result["status"] == "success":
                        st.session_state.df = result["data"]
                        st.session_state.dataset_name = sample_name
                        st.session_state.eda_results = {}
                        st.session_state.analysis_results = {}
                        st.session_state.plot_history = []
                        st.session_state.report = None
                        st.success(f"Active dataset: {sample_name}")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"**Active Dataset:** `{st.session_state.dataset_name}`")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Rows", df.shape[0])
        with col_s2:
            st.metric("Columns", df.shape[1])

        if st.button("🗑️ Reset Dataset"):
            st.session_state.df = None
            st.session_state.dataset_name = None
            st.session_state.eda_results = {}
            st.session_state.analysis_results = {}
            st.session_state.plot_history = []
            st.session_state.report = None
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("## ⚙️ Multi-Agent Settings")
    
    backend_mode = st.selectbox(
        "Orchestrator Backend",
        [
            "⚡ Built-in Semantic Parser (Fast & Cloud-Ready)",
            "🤗 Hugging Face Inference API (Serverless)",
            "💻 Local Mistral-7B (Requires CUDA GPU)"
        ],
        index=0
    )

    hf_token_input = st.text_input(
        "Hugging Face Token (optional)",
        value=HF_TOKEN if HF_TOKEN else "",
        type="password",
        help="Used when Hugging Face Inference API or gated models are selected."
    )
    if hf_token_input:
        os.environ["HF_TOKEN"] = hf_token_input

    st.markdown("---")
    st.markdown("## 💡 Quick Query Shortcuts")
    st.caption("Click any shortcut to trigger specialized agents:")

    shortcuts = [
        ("🔍 Run Full EDA", "perform exploratory data analysis and summarize statistics"),
        ("📊 Scatter Plot", "generate a scatter plot"),
        ("📈 Histogram Distribution", "show histogram distribution"),
        ("📦 Box Plot (Outliers)", "show box plot to check outliers"),
        ("🔥 Correlation Heatmap", "show correlation heatmap"),
        ("📑 Full Data Science Report", "generate a comprehensive data science report")
    ]

    for label, query_text in shortcuts:
        if st.button(label, use_container_width=True):
            st.session_state.pending_query = query_text
            st.rerun()

# Execution Pipeline Function
def process_query(query: str):
    if not query.strip():
        return

    context = {
        "df": st.session_state.df,
        "eda_results": st.session_state.eda_results,
        "analysis_results": st.session_state.analysis_results,
        "plot_results": [p["caption"] for p in st.session_state.plot_history]
    }

    # Step 1: Run Orchestrator Agent
    orch_output = orchestrator_agent(query, context)
    target_agent = orch_output.target_agent
    parameters = orch_output.parameters
    orch_message = orch_output.message

    agent_response_lines = [f"**Orchestrator Decision:** {orch_message}"]

    if target_agent == "data":
        if st.session_state.df is None:
            agent_response_lines.append("⚠️ **Data Agent:** Please upload a CSV dataset or pick a sample dataset from the sidebar first.")
        else:
            agent_response_lines.append(f"✅ **Data Agent:** Dataset `{st.session_state.dataset_name}` is already active with {st.session_state.df.shape[0]} rows and {st.session_state.df.shape[1]} columns.")

    elif target_agent == "eda":
        if st.session_state.df is None or st.session_state.df.empty:
            agent_response_lines.append("⚠️ **EDA Agent:** No dataset loaded. Please upload a dataset from the sidebar.")
        else:
            result = perform_eda(st.session_state.df)
            st.session_state.eda_results = result
            agent_response_lines.append(f"📊 **EDA Agent:** {result['message']}")
            if result["status"] == "success":
                agent_response_lines.append("### Key Insights Summary:")
                for col, stats_vals in result["summary_stats"].items():
                    if isinstance(stats_vals, dict):
                        mean_val = stats_vals.get('mean')
                        mean_str = f"{mean_val:.2f}" if isinstance(mean_val, (int, float)) else "N/A"
                        missing_cnt = result['missing_values'].get(col, 0)
                        outlier_cnt = result['outliers'].get(col, 0)
                        agent_response_lines.append(f"- **{col}**: Mean = `{mean_str}`, Missing = `{missing_cnt}`, Outliers = `{outlier_cnt}`")

    elif target_agent == "visualization":
        if st.session_state.df is None or st.session_state.df.empty:
            agent_response_lines.append("⚠️ **Visualization Agent:** Please upload a dataset first to generate plots.")
        else:
            result = generate_plot(st.session_state.df, **parameters)
            agent_response_lines.append(f"🎨 **Visualization Agent:** {result['message']}")
            if result["status"] == "success":
                plot_file = result.get("file")
                if not plot_file or not os.path.exists(plot_file):
                    plot_files = sorted(glob.glob(f"plot_{parameters.get('plot_type', 'auto')}*.png"), key=os.path.getmtime)
                    if plot_files:
                        plot_file = plot_files[-1]

                if plot_file and os.path.exists(plot_file):
                    ptype = parameters.get('plot_type', 'Visualization').capitalize()
                    xcol = parameters.get('x_col', '')
                    ycol = parameters.get('y_col')
                    vs_str = f" vs {ycol}" if ycol else ""
                    caption = f"{ptype} Plot ({xcol}{vs_str})".strip()
                    st.session_state.plot_history.append({
                        "file": plot_file,
                        "caption": caption,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    agent_response_lines.append("✅ Generated plot saved and added to Visualizations tab.")
            else:
                agent_response_lines.append(f"❌ Error: {result.get('message', 'Plot generation failed.')}")

    elif target_agent == "analysis":
        if st.session_state.df is None or st.session_state.df.empty:
            agent_response_lines.append("⚠️ **Analysis Agent:** Please upload a dataset first.")
        else:
            result = calculate_correlation(st.session_state.df, **parameters)
            st.session_state.analysis_results = result
            if result["status"] == "success":
                corr_val = result.get('correlation')
                corr_str = f"{corr_val:.4f}" if corr_val is not None else "N/A"
                agent_response_lines.append(f"📈 **Analysis Agent:** Correlation between **{result.get('col1', 'Col1')}** and **{result.get('col2', 'Col2')}**: `{corr_str}`")
                agent_response_lines.append("#### Full Correlation Matrix calculated. View details in the EDA & Statistics tab.")
            else:
                agent_response_lines.append(f"⚠️ **Analysis Agent:** {result.get('message')}")

    elif target_agent == "report":
        if st.session_state.df is None or st.session_state.df.empty:
            agent_response_lines.append("⚠️ **Report Agent:** No data loaded. Please upload a dataset first.")
        else:
            if not st.session_state.eda_results:
                st.session_state.eda_results = perform_eda(st.session_state.df)
                agent_response_lines.append("ℹ️ Automatically ran EDA Agent for report.")
            if not st.session_state.analysis_results:
                st.session_state.analysis_results = calculate_correlation(st.session_state.df)
                agent_response_lines.append("ℹ️ Automatically ran Correlation Analysis Agent for report.")
            if not st.session_state.plot_history:
                plot_result = generate_plot(st.session_state.df)
                if plot_result["status"] == "success" and plot_result.get("file"):
                    st.session_state.plot_history.append({
                        "file": plot_result["file"],
                        "caption": "Auto-generated Overview Plot",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                agent_response_lines.append("ℹ️ Automatically ran Visualization Agent for default plot.")

            plot_captions = [p["caption"] for p in st.session_state.plot_history]
            rep_res = generate_report(
                st.session_state.df,
                st.session_state.eda_results,
                st.session_state.analysis_results,
                plot_captions
            )
            st.session_state.report = rep_res.get("report", "")
            agent_response_lines.append("📄 **Report Agent:** Comprehensive Data Science Report generated successfully! Check the **Report Tab**.")

    # Record in history
    st.session_state.chat_history.append({
        "query": query,
        "target_agent": target_agent,
        "parameters": parameters,
        "response": "\n\n".join(agent_response_lines),
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

# Main Header
st.markdown('<div class="main-title">Multi-Agent System for Data Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Automated Data Ingestion, EDA, Statistical Analysis, Visualization & Report Generation powered by specialized AI agents.</div>',
    unsafe_allow_html=True
)

# Agent Badges Bar
st.markdown("""
<div>
    <span class="agent-badge badge-viz">🎯 Central Orchestrator</span>
    <span class="agent-badge badge-data">📥 Data Ingestion</span>
    <span class="agent-badge badge-eda">🔍 Exploratory Data Analysis</span>
    <span class="agent-badge badge-analysis">📈 Statistical Correlation</span>
    <span class="agent-badge badge-viz">🎨 Visualization</span>
    <span class="agent-badge badge-report">📑 Report Generation</span>
</div>
""", unsafe_allow_html=True)

st.write("")

# Query Input
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_query = st.text_input(
        "Enter your instruction or query for the multi-agent system:",
        placeholder="e.g. 'Perform EDA', 'Scatter plot total_bill vs tip', 'Correlation matrix', 'Generate full report'...",
        key="query_input_field"
    )
with col_btn:
    st.write("")
    submit_clicked = st.button("🚀 Run Agents", use_container_width=True)

# Trigger query execution from text input or pending shortcut
query_to_run = None
if submit_clicked and user_query:
    query_to_run = user_query
elif st.session_state.pending_query:
    query_to_run = st.session_state.pending_query
    st.session_state.pending_query = None

if query_to_run:
    with st.spinner("🤖 Multi-Agent system coordinating..."):
        process_query(query_to_run)

# Dashboard Tabs
tab_chat, tab_viz, tab_eda, tab_report, tab_raw = st.tabs([
    "💬 Agent Activity & Output",
    "🎨 Visualizations Gallery",
    "🔍 Exploratory Data Analysis",
    "📑 Comprehensive Report",
    "📋 Raw Dataset Explorer"
])

# Tab 1: Agent Activity & Chat
with tab_chat:
    if not st.session_state.chat_history:
        st.info("👋 Welcome! Upload a dataset or choose a sample in the sidebar, then type a command or click a quick shortcut.")
    else:
        for idx, entry in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"#### 👤 Query: *\"{entry['query']}\"* &nbsp; `[{entry['timestamp']}]`")
                badge_html = get_agent_badge(entry['target_agent'])
                st.markdown(f"**Target Agent:** {badge_html}", unsafe_allow_html=True)
                
                with st.expander("🛠️ Orchestrator Execution Details", expanded=False):
                    st.json({
                        "target_agent": entry["target_agent"],
                        "parameters": entry["parameters"],
                    })

                st.markdown(entry["response"])
                st.markdown("---")

# Tab 2: Visualizations
with tab_viz:
    if not st.session_state.plot_history:
        st.info("No plots generated yet. Try asking: *'Generate scatter plot'* or *'Show correlation heatmap'*.")
    else:
        st.markdown(f"### Generated Visualizations ({len(st.session_state.plot_history)})")
        cols = st.columns(2)
        for i, item in enumerate(reversed(st.session_state.plot_history)):
            plot_file = item["file"]
            if os.path.exists(plot_file):
                with cols[i % 2]:
                    st.image(plot_file, caption=f"{item['caption']} ({item['timestamp']})", use_container_width=True)
                    with open(plot_file, "rb") as f:
                        st.download_button(
                            label=f"💾 Download {os.path.basename(plot_file)}",
                            data=f.read(),
                            file_name=os.path.basename(plot_file),
                            mime="image/png",
                            key=f"dl_plot_{i}"
                        )
                    st.write("")

# Tab 3: Exploratory Data Analysis (EDA)
with tab_eda:
    if st.session_state.df is None:
        st.info("Please load a dataset to view EDA metrics.")
    elif not st.session_state.eda_results:
        st.info("EDA has not been performed yet. Click **'🔍 Run Full EDA'** in the sidebar or ask **'Perform EDA'**.")
    else:
        eda = st.session_state.eda_results
        st.markdown("### 📊 Dataset Overview & Statistical Summary")
        
        # Summary Stats Table
        if "summary_stats" in eda and eda["summary_stats"]:
            stats_df = pd.DataFrame(eda["summary_stats"])
            st.markdown("#### Summary Statistics")
            st.dataframe(stats_df, use_container_width=True)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if "missing_values" in eda and eda["missing_values"]:
                st.markdown("#### Missing Values per Column")
                miss_df = pd.DataFrame(list(eda["missing_values"].items()), columns=["Column", "Missing Count"])
                st.dataframe(miss_df, use_container_width=True)

        with col_e2:
            if "outliers" in eda and eda["outliers"]:
                st.markdown("#### Outliers Detected (IQR Method)")
                out_df = pd.DataFrame(list(eda["outliers"].items()), columns=["Column", "Outlier Count"])
                st.dataframe(out_df, use_container_width=True)

        if "corr_matrix" in eda and eda["corr_matrix"]:
            st.markdown("#### Correlation Matrix")
            corr_df = pd.DataFrame(eda["corr_matrix"])
            st.dataframe(corr_df.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1), use_container_width=True)

# Tab 4: Comprehensive Report
with tab_report:
    if not st.session_state.report:
        st.info("No report generated yet. Click **'📑 Full Data Science Report'** or ask **'Generate report'** to create one.")
    else:
        st.markdown("### 📑 Automated Data Science Report")
        st.download_button(
            label="📥 Download Report (.md)",
            data=st.session_state.report,
            file_name=f"data_science_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
        st.markdown(f"```text\n{st.session_state.report}\n```")

# Tab 5: Raw Dataset Explorer
with tab_raw:
    if st.session_state.df is None:
        st.info("No dataset is currently active. Upload a CSV or select a sample dataset from the sidebar.")
    else:
        st.markdown(f"### Active Dataset: `{st.session_state.dataset_name}`")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Rows", st.session_state.df.shape[0])
        with col_m2:
            st.metric("Total Columns", st.session_state.df.shape[1])
        with col_m3:
            st.metric("Memory Usage", f"{st.session_state.df.memory_usage().sum() / 1024:.2f} KB")

        st.dataframe(st.session_state.df, use_container_width=True)

        st.markdown("#### Column Data Types")
        dtypes_df = pd.DataFrame({
            "Column": st.session_state.df.columns,
            "Data Type": [str(t) for t in st.session_state.df.dtypes]
        })
        st.dataframe(dtypes_df, use_container_width=True)
