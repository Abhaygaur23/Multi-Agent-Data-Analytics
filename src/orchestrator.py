import re
import pandas as pd
from src.agent_outputs import OrchestratorOutput
from src.utils import extract_json
from src.models import call_llm

def fallback_intent_detection(query: str, has_data: bool) -> tuple:
    q = query.lower()
    if any(k in q for k in ["eda", "explor", "summar", "describe", "missing", "skew", "outlier", "stats"]):
        return "eda", "Identified intent as Exploratory Data Analysis."
    if any(k in q for k in ["scatter", "histogram", "hist", "line", "box", "count", "heatmap", "plot", "chart", "visual", "graph"]):
        return "visualization", "Identified intent as Data Visualization."
    if any(k in q for k in ["correlation", "corr", "matrix", "relationship", "association"]):
        return "analysis", "Identified intent as Statistical / Correlation Analysis."
    if any(k in q for k in ["report", "data science report", "summary report", "briefing"]):
        return "report", "Identified intent as Comprehensive Report Generation."
    if any(k in q for k in ["load", "clean", "upload", "import", "read", "dataset", "csv"]):
        return "data", "Identified intent as Data Ingestion and Cleaning."
    return "eda" if has_data else "data", "Routed based on query context."

# Implement the Orchestrator Agent
def orchestrator_agent(query: str, context: dict) -> OrchestratorOutput:
    df = context.get("df") if context else None
    has_data = df is not None and isinstance(df, pd.DataFrame) and not df.empty

    prompt = (
        f"You are a data science orchestration system. Based on the user query, determine which "
        f"specialized agent should handle the request.\n\n"
        f"User query: \"{query}\"\n"
        f"Data loaded: {'Yes' if has_data else 'No'}\n\n"
        f"Available agents:\n"
        f"- 'data': For loading datasets\n"
        f"- 'eda': For exploratory data analysis\n"
        f"- 'visualization': For creating plots\n"
        f"- 'analysis': For statistical analysis\n"
        f"- 'report': For generating reports (automatically perform prior steps if needed)\n\n"
        f"Respond with a JSON object: {{\"target_agent\": \"name\", \"message\": \"explanation\"}}"
    )

    target_agent = "none"
    message = "Processing request..."

    try:
        llm_response = call_llm(prompt)
        json_response = extract_json(llm_response)
        if json_response and "target_agent" in json_response:
            target_agent = str(json_response.get("target_agent", "")).lower().strip()
            message = str(json_response.get("message", f"Routing to {target_agent} agent."))
    except Exception as e:
        json_response = {}

    if target_agent not in ["data", "eda", "visualization", "analysis", "report"]:
        target_agent, message = fallback_intent_detection(query, has_data)

    parameters = {}
    df_cols = list(df.columns) if has_data else []

    if target_agent == "data":
        file_path_match = re.search(r'([A-Za-z0-9_\-\\/:\.]+\.csv)', query, re.IGNORECASE)
        if file_path_match:
            parameters["file_path"] = file_path_match.group(1)
            
    elif target_agent == "visualization":
        # Extract plot type from query
        plot_type = "auto"
        query_lower = query.lower()
        if "scatter" in query_lower:
            plot_type = "scatter"
        elif "line" in query_lower:
            plot_type = "line"
        elif "histogram" in query_lower or "hist" in query_lower:
            plot_type = "histogram"
        elif "box" in query_lower:
            plot_type = "box"
        elif "count" in query_lower:
            plot_type = "count"
        elif "heatmap" in query_lower:
            plot_type = "heatmap"

        parameters["plot_type"] = plot_type

        # Extract column names from query (case-insensitive)
        columns = []
        for col in df_cols:
            if str(col).lower() in query_lower:
                columns.append(col)

        if len(columns) >= 1:
            parameters["x_col"] = columns[0]
        if len(columns) >= 2:
            parameters["y_col"] = columns[1]

        # If no columns specified but plot type requires them, use defaults if data exists
        if has_data and not parameters.get("x_col") and plot_type in ["scatter", "line", "histogram", "box", "count"]:
            numeric_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)
            parameters["x_col"] = numeric_cols[0] if len(numeric_cols) > 0 else (df_cols[0] if df_cols else None)
            if plot_type in ["scatter", "line"] and len(numeric_cols) > 1:
                parameters["y_col"] = numeric_cols[1]

    elif target_agent == "analysis":
        # Extract column names from query for correlation
        query_lower = query.lower()
        columns = [col for col in df_cols if str(col).lower() in query_lower]
        if len(columns) >= 2:
            parameters["col1"] = columns[0]
            parameters["col2"] = columns[1]

    return OrchestratorOutput(status="success", message=message, target_agent=target_agent, parameters=parameters)
