import sys
import os

# Check if running inside Streamlit
try:
    import streamlit as st
    is_streamlit = st.runtime.exists()
except Exception:
    is_streamlit = False

if is_streamlit:
    # Run Streamlit App
    import streamlit_app
else:
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Data Analytics Application")
    parser.add_argument("--gradio", action="store_true", help="Launch the legacy Gradio interface instead of Streamlit")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
    args, unknown = parser.parse_known_args()

    if args.gradio:
        import gradio as gr
        from src.tools import load_and_clean_data, perform_eda, generate_plot, calculate_correlation, generate_report
        from src.orchestrator import orchestrator_agent
        from src.config import DEVICE

        # Gradio inference function
        def run_data_science_assistant_gradio(query, file=None):
            context = {"df": None, "eda_results": {}, "analysis_results": {}, "plot_results": []}
            output_text = []
            output_images = []

            if file is not None:
                file_path = file.name
                result = load_and_clean_data(file_path)
                if result["status"] == "success":
                    context["df"] = result["data"]
                    output_text.append(f"Data Agent: {result['message']}")
                else:
                    output_text.append(f"Data Agent: {result['message']}")
                    return "\n".join(output_text), []

            if query:
                orch_output = orchestrator_agent(query, context)
                output_text.append(f"Orchestrator: {orch_output.message}")
                target_agent = orch_output.target_agent
                parameters = orch_output.parameters

                if target_agent == "data":
                    if file is None:
                        output_text.append("Assistant: Please upload a dataset file first.")

                elif target_agent == "eda":
                    if context["df"] is None or context["df"].empty:
                        output_text.append("Assistant: Please upload a dataset first.")
                    else:
                        result = perform_eda(context["df"])
                        context["eda_results"] = result
                        output_text.append(f"Assistant: {result['message']}")
                        if result["status"] == "success":
                            output_text.append("Key Insights:")
                            for col, stats in result["summary_stats"].items():
                                if isinstance(stats, dict):
                                    output_text.append(f"  {col}: mean={stats.get('mean', 'N/A')}, missing={result['missing_values'].get(col, 0)}")

                elif target_agent == "visualization":
                    if context["df"] is None or context["df"].empty:
                        output_text.append("Assistant: Please upload a dataset first.")
                    else:
                        result = generate_plot(context["df"], **parameters)
                        context["plot_results"].append(result["message"])
                        output_text.append(f"Assistant: {result['message']}")
                        if result["status"] == "success":
                            import glob
                            plot_files = sorted(glob.glob(f"plot_{parameters.get('plot_type', 'auto')}*.png"), key=os.path.getmtime)
                            if plot_files:
                                output_images.append(plot_files[-1])

                elif target_agent == "analysis":
                    if context["df"] is None or context["df"].empty:
                        output_text.append("Assistant: Please upload a dataset first.")
                    else:
                        result = calculate_correlation(context["df"], **parameters)
                        context["analysis_results"] = result
                        if result["status"] == "success":
                            output_text.append(f"Assistant: Correlation between {result['col1']} and {result['col2']} is {result['correlation']:.4f}")
                            output_text.append("Full Correlation Matrix:")
                            for col1, corrs in result["corr_matrix"].items():
                                for col2, corr in corrs.items():
                                    output_text.append(f"  {col1} vs {col2}: {corr:.4f}")
                        else:
                            output_text.append(f"Assistant: {result['message']}")

                elif target_agent == "report":
                    if context["df"] is None or context["df"].empty:
                        output_text.append("Assistant: No data loaded. Please upload a dataset first.")
                    else:
                        if not context["eda_results"]:
                            context["eda_results"] = perform_eda(context["df"])
                            output_text.append("Assistant: Performed EDA for report.")
                        if not context["analysis_results"]:
                            context["analysis_results"] = calculate_correlation(context["df"])
                            output_text.append("Assistant: Performed correlation analysis for report.")
                        if not context["plot_results"]:
                            plot_result = generate_plot(context["df"])
                            context["plot_results"].append(plot_result["message"])
                            output_text.append("Assistant: Generated a default plot for report.")
                            import glob
                            plot_files = glob.glob("plot_auto*.png")
                            if plot_files:
                                output_images.append(plot_files[-1])
                        result = generate_report(context["df"], context["eda_results"], context["analysis_results"], context["plot_results"])
                        output_text.append(f"Assistant: {result['message']}")
                        output_text.append(result["report"])

            return "\n".join(output_text), output_images

        with gr.Blocks(title="Multi-Agent System for Data Analysis") as demo:
            gr.Markdown("# Multi-Agent System for Data Analysis (Gradio)")
            gr.Markdown("Upload a dataset and enter a query to perform data science tasks.")
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload Dataset (CSV)")
                    query_input = gr.Textbox(label="Enter your query", placeholder="e.g., 'perform EDA' or 'scatter plot'")
                    submit_btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output", lines=20)
                    output_gallery = gr.Gallery(label="Visualizations")

            submit_btn.click(
                fn=run_data_science_assistant_gradio,
                inputs=[query_input, file_input],
                outputs=[output_text, output_gallery]
            )

        demo.launch(share=False)
    else:
        # Default: Launch Streamlit
        print("Launching Streamlit app...")
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", str(args.port)])
