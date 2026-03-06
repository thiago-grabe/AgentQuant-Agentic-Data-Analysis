import os
import asyncio
import pandas as pd
import json
import logging
import traceback
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions import KernelArguments

os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# -----------------
# Logging Setup
# -----------------
# The logging setup below captures all agent interactions and saves them to 'logs/agent_chat.log'.
# 1. Create a dedicated logger for agent interactions.
agent_logger = logging.getLogger("semantic_kernel.agents")
agent_logger.setLevel(logging.DEBUG)

# 2. Prevent agent logs from propagating to other handlers (like console).
agent_logger.propagate = False

# 3. Create a file handler to write to 'agent_chat.log' in write mode.
agent_chat_handler = logging.FileHandler("logs/agent_chat.log", mode='w')
agent_chat_handler.setLevel(logging.DEBUG)

# 4. Create a minimal formatter to log only the message content.
chat_formatter = logging.Formatter('%(asctime)s - %(name)s:%(message)s')
agent_chat_handler.setFormatter(chat_formatter)

# 5. Add the dedicated file handler to the agent logger.
agent_logger.addHandler(agent_chat_handler)

# 6. Function to log agent messages
def log_agent_message(content):
    try:
        agent_logger.info(f"Agent: {content.role} - {content.name or '*'}: {content.content}")
    except Exception:
        agent_logger.exception("Failed to write agent message to log")

# -----------------
# Environment Setup
# -----------------
load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_KEY")
BASE_URL = os.getenv("URL")
API_VERSION = "2024-05-01-preview"


# -----------------
# Kernel and Chat Service
# -----------------
kernel = Kernel()
chat_service = AzureChatCompletion(
    api_key=API_KEY,
    endpoint=BASE_URL,
    deployment_name="gpt-4.1",
    api_version=API_VERSION,
)
kernel.add_service(chat_service)


# -----------------
# Helper Functions
# -----------------
# <TODO: Step 4 - Implement Supporting Logic>
# Implement the logic for each of the helper functions below.

def load_quality_instructions(file_path):
    """
    Loads instructional text from a file within the 'specs' directory.

    This function constructs the full path to the file, reads its content,
    and processes it into a list of non-empty, stripped lines.

    Args:
        file_path (str): The name of the file in the 'specs' directory.

    Returns:
        list[str]: A list of strings, where each string is a line of instruction.
                   Returns an empty list if the file does not exist.
    """
    full_path = os.path.join("specs", file_path)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {full_path} not found.")
        return []

def load_reports_instructions(file_path):
    """
    Loads report generation instructions from a file within the 'specs' directory.

    Args:
        file_path (str): The name of the file in the 'specs' directory.

    Returns:
        list[str]: A list of strings for building the report. Returns an
                   empty list if the file does not exist.
    """
    full_path = os.path.join("specs", file_path)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {full_path} not found.")
        return []

def load_logs(file_path):
    """
    Loads agent interaction logs from a file within the 'logs' directory.

    Args:
        file_path (str): The name of the log file in the 'logs' directory.

    Returns:
        list[str]: A list of log entries. Returns an empty list if the file
                   does not exist.
    """
    full_path = os.path.join("logs", file_path)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {full_path} not found.")
        return []

def get_csv_name():
    """
    Interactively prompts the user to select a CSV file from the 'data' directory.

    It lists all available .csv files and asks for a numerical selection.

    Returns:
        str: The relative path to the selected CSV file (e.g., 'data/my_file.csv').
    """
    data_dir = "data"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the data/ directory.")
    csv_files.sort()
    print("\nAvailable CSV files:")
    for i, fname in enumerate(csv_files, 1):
        print(f"  {i}. {fname}")
    while True:
        try:
            choice = int(input("\nSelect a file by number: "))
            if 1 <= choice <= len(csv_files):
                selected = os.path.join(data_dir, csv_files[choice - 1])
                print(f"Selected: {selected}")
                return selected
            print(f"Please enter a number between 1 and {len(csv_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def load_csv_file(file_path):
    """
    Reads a CSV file and converts its entire content into a single string.

    The CSV data is flattened into a list and then joined by ', '.

    Args:
        file_path (str): The path to the CSV file to load.

    Returns:
        str: A single string containing all the data from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        all_data = list(df.columns) + df.values.flatten().tolist()
        return ', '.join(str(item) for item in all_data)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return ""

class PythonExecutor:
    """
    A safe executor for dynamically generated Python code strings.

    This class is designed to run code provided by an AI agent in a controlled
    manner. It includes a retry mechanism and captures execution errors.
    """
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts

    def run(self, code):
        """
        Executes a string of Python code using the exec() function.

        Args:
            code (str): The Python code to execute.

        Returns:
            tuple[bool, str | None]: A tuple containing:
                - A boolean indicating if the execution was successful.
                - The error traceback as a string if an exception occurred,
                  otherwise None.
        """
        try:
            exec(code)
            return True, None
        except Exception:
            return False, traceback.format_exc()

def save_final_report(report, path='artifacts/final_report.md'):
    """
    Saves the generated final report to a markdown file.

    Args:
        report (str): The content of the report to be saved.
        path (str, optional): The file path for the saved report.
                              Defaults to 'artifacts/final_report.md'.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {path}")
    except Exception as e:
        print(f"Error saving report: {e}")


# -----------------
# Agent Instructions
# -----------------
# <TODO: Step 5 - Build the Agents and Teams>
# 1. Complete the AGENT_CONFIG with detailed prompts for each agent.
data_quality_instructions = '\n'.join(load_quality_instructions("Data_Quality_Instructions.txt"))
report_instructions = '\n'.join(load_reports_instructions("Report_Instructions.txt"))

AGENT_CONFIG = {
    "PythonExecutorAgent": '''You are a Python Code Generator. Your sole purpose is to generate executable Python code for data visualization.

Rules you MUST follow:
1. Use matplotlib to create a single line chart that plots BOTH the original data (in blue, label="Original") and the cleaned data (in green, label="Cleaned") on the same axes.
2. Add a title, axis labels, a legend, and a grid to the chart.
3. Before saving, ensure the "artifacts" directory exists using os.makedirs("artifacts", exist_ok=True).
4. Save the plot to "artifacts/data_visualization.png" using plt.savefig().
5. Call plt.close() after saving to free memory.
6. Your output must contain ONLY the raw, runnable Python code. No explanations, no comments, no markdown fences, no surrounding text.
7. Import all necessary libraries (matplotlib.pyplot, os) at the top of your code.
8. Parse the data provided in the prompt to extract dates/indices and values for both original and cleaned datasets.''',

    "DataCleaning": '''You are a Data Cleaning Assistant. Your job is to clean raw datasets by identifying and removing outliers and erroneous values.

Follow this process step by step:
1. First, examine the data and present your cleaning plan, explaining which values you consider outliers and why (e.g., zero values that represent missing data, extreme values far from the mean).
2. Apply your cleaning plan by removing the identified outliers.
3. Present the final cleaned dataset clearly, listing each data point (date/index and value).

Your final output after cleaning must be ONLY the cleaned data in a clear tabular or list format. No additional commentary after the cleaned data.''',

    "DataStatistics": '''You are a Data Statistics Assistant. Your job is to compute descriptive statistics on the data provided to you.

When given cleaned data, compute and present ONLY the following statistics:
- Count: number of data points
- Mean: average value
- Median: middle value
- Standard Deviation: measure of spread
- Minimum: smallest value
- Maximum: largest value

Present the statistics in a clean table format. Output ONLY the statistical description with no additional commentary or explanation.''',

    "AnalysisChecker": f'''You are a Data Validation Auditor. Your job is to verify the quality of the data cleaning and statistical analysis performed by other agents.

Follow these validation rules:
{data_quality_instructions}

Your process:
1. Check that ALL identified outliers have been removed from the cleaned dataset.
2. Verify that the descriptive statistics (mean, median, standard deviation, minimum, maximum) were computed using ONLY the cleaned dataset, not the original data with outliers.
3. If BOTH checks pass, output "Approved" as the title.
4. If EITHER check fails, output a clear error message explaining exactly what went wrong.

Your output MUST be in JSON format with these fields:
- "title": "Approved" or "Failed"
- "original_data": table of the original dataset
- "cleaned_data": table of the cleaned dataset
- "removed_data": table of removed outliers
- "descriptive_statistics": table of computed statistics''',

    "ReportGenerator": f'''You are a Report Generator. Your job is to synthesize all data analysis results into a comprehensive, well-structured markdown report.

Follow this exact report structure:
{report_instructions}

Instructions:
1. Fill in ALL sections of the template using the actual data from the analysis — cleaning results, statistics, validation outcomes, and visualization.
2. Include the data date range derived from the dataset.
3. In the Data Cleaning section, describe the approach, list removed outliers in a table, and show the cleaned data in a table.
4. In the Descriptive Statistics section, present all computed statistics in a table and add a brief interpretive summary.
5. In the Validation Summary, describe each iteration of checking.
6. Reference the visualization image as: ![Data Visualization](data_visualization.png)
7. Write clear conclusions about the data quality and analysis findings.
8. Complete the Agent Workflow Summary table with all agents and their actions.
9. Output ONLY the complete markdown report.''',

    "ReportChecker": f'''You are a Report Validation Auditor. Your job is to review the generated report for completeness, accuracy, and proper formatting.

Validate the report against these requirements:
{report_instructions}

Check each of the following:
1. The report has a title "# Data Analysis Report" and a Data Date field.
2. ALL required sections exist: Overview, Data Cleaning, Descriptive Statistics, Validation Summary, Data Visualization, Conclusions, and Agent Workflow Summary.
3. The Data Cleaning section includes the approach, a table of removed outliers, a table of cleaned data, and a result summary.
4. The Descriptive Statistics section includes a table with count, mean, median, std, min, max and a summary.
5. The Validation Summary describes at least one iteration.
6. The Data Visualization section references the image file.
7. The Conclusions section provides meaningful analysis.
8. The Agent Workflow Summary table lists all agents with their steps, actions, and status.

If the report passes ALL checks, output "Approved: The report is complete and accurate."
If any check fails, describe exactly what is missing or incorrect so the ReportGenerator can fix it.'''
}


# -----------------
# Agent Factory
# -----------------
# <TODO: Step 5 - Build the Agents and Teams>
# 2. Implement the agent factory function.
def create_agent(name, instructions, service, settings=None):
    """Factory function to create a new ChatCompletionAgent."""
    kwargs = {
        "service": service,
        "name": name,
        "instructions": instructions,
    }
    if settings:
        kwargs["arguments"] = KernelArguments(settings=settings)
    return ChatCompletionAgent(**kwargs)


# -----------------
# Termination Strategy
# -----------------
# A custom termination strategy that stops after user approval.
class ApprovalTerminationStrategy(TerminationStrategy):
    """A custom termination strategy that stops after user approval."""
    async def should_agent_terminate(self, agent, history):
        if history and "approved" in history[-1].content.lower():
            return True
        return False


# -----------------
# Agent Instantiation
# -----------------
# <TODO: Step 5 - Build the Agents and Teams>
# 3. Instantiate each agent with the correct name, prompt, and temperature setting.
python_agent = create_agent(
    "PythonExecutorAgent",
    AGENT_CONFIG["PythonExecutorAgent"],
    chat_service,
    OpenAIChatPromptExecutionSettings(temperature=0.1),
)
cleaning_agent = create_agent(
    "DataCleaning",
    AGENT_CONFIG["DataCleaning"],
    chat_service,
    OpenAIChatPromptExecutionSettings(temperature=0.7),
)
stats_agent = create_agent(
    "DataStatistics",
    AGENT_CONFIG["DataStatistics"],
    chat_service,
    OpenAIChatPromptExecutionSettings(temperature=0.5),
)
checker_agent = create_agent(
    "AnalysisChecker",
    AGENT_CONFIG["AnalysisChecker"],
    chat_service,
    OpenAIChatPromptExecutionSettings(temperature=0.2),
)
report_agent = create_agent(
    "ReportGenerator",
    AGENT_CONFIG["ReportGenerator"],
    chat_service,
    OpenAIChatPromptExecutionSettings(temperature=1.0),
)
report_checker_agent = create_agent(
    "ReportChecker",
    AGENT_CONFIG["ReportChecker"],
    chat_service,
    OpenAIChatPromptExecutionSettings(temperature=0.2),
)


# -----------------
# Group Chats
# -----------------
# <TODO: Step 5 - Build the Agents and Teams>
# 4. Create the three agent group chats.
analysis_chat = AgentGroupChat(
    agents=[cleaning_agent, stats_agent, checker_agent],
    termination_strategy=ApprovalTerminationStrategy(
        agents=[checker_agent],
        maximum_iterations=10,
    ),
)

code_chat = AgentGroupChat(
    agents=[python_agent],
)

report_chat = AgentGroupChat(
    agents=[report_agent, report_checker_agent],
)


# -----------------
# Main Workflow
# -----------------
# <TODO: Step 6 - Orchestrate the Main Workflow>
# Implement the main workflow logic, following the sequence described in the instructions.
def extract_code(text):
    """Strip markdown code fences from agent-generated code."""
    if "```python" in text:
        text = text.split("```python", 1)[1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()


async def main():
    """The main entry point for the agentic workflow."""

    # ---- Phase 1: Data Ingestion & Analysis ----

    # 1. Load the CSV data.
    csv_path = get_csv_name()
    csv_data = load_csv_file(csv_path)
    if not csv_data:
        print("Failed to load CSV data. Exiting.")
        return

    initial_prompt = (
        f"Here is the raw CSV data to analyze. The columns and values are: {csv_data}\n\n"
        "Please clean this data by identifying and removing outliers, then compute descriptive statistics."
    )
    print("\n========== Phase 1: Data Analysis ==========")

    # 2. Invoke the analysis chat.
    await analysis_chat.add_chat_message(
        ChatMessageContent(role=AuthorRole.USER, content=initial_prompt)
    )

    analysis_result = None
    async for content in analysis_chat.invoke():
        log_agent_message(content)
        print(f"\n[{content.name}]:\n{content.content}")
        analysis_result = content.content

    # 3. Get human approval.
    print("\n========== Human Approval Required ==========")
    approval = input("Do you approve the analysis results? (yes/no): ").strip().lower()
    if approval != "yes":
        print("Analysis not approved. Workflow terminated.")
        return
    print("Analysis approved. Proceeding to visualization...\n")

    # 4. Save the cleaned data.
    cleaned_data_path = "data-cleaned.json"
    try:
        with open(cleaned_data_path, "w", encoding="utf-8") as f:
            json.dump({"analysis_result": analysis_result}, f, indent=2)
        print(f"Cleaned data saved to {cleaned_data_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")

    # ---- Phase 2: Code Generation & Execution ----
    print("\n========== Phase 2: Code Generation ==========")

    # 5. Invoke the code chat to generate visualization code.
    code_prompt = (
        f"Generate Python code to visualize the following data.\n\n"
        f"Original raw data (columns and all values): {csv_data}\n\n"
        f"Cleaned/analyzed data and statistics:\n{analysis_result}\n\n"
        "Plot both original data (blue line, label='Original') and cleaned data "
        "(green line, label='Cleaned') on the same matplotlib line chart. "
        "Save the plot to 'artifacts/data_visualization.png'."
    )

    await code_chat.add_chat_message(
        ChatMessageContent(role=AuthorRole.USER, content=code_prompt)
    )

    generated_code = ""
    async for content in code_chat.invoke():
        log_agent_message(content)
        generated_code = content.content

    # 6. Execute the code in a retry loop.
    executor = PythonExecutor(max_attempts=10)
    code = extract_code(generated_code)
    success = False

    for attempt in range(executor.max_attempts):
        print(f"\nCode execution attempt {attempt + 1}/{executor.max_attempts}...")
        ok, error = executor.run(code)
        if ok:
            print("Code executed successfully!")
            success = True
            break
        print(f"Execution failed: {error}")
        code_chat.is_complete = False
        error_prompt = (
            f"The code failed with this error:\n{error}\n\n"
            "Please fix the code and output ONLY the corrected Python code."
        )
        await code_chat.add_chat_message(
            ChatMessageContent(role=AuthorRole.USER, content=error_prompt)
        )
        async for content in code_chat.invoke():
            log_agent_message(content)
            generated_code = content.content
        code = extract_code(generated_code)

    if not success:
        print("Failed to generate working visualization code after all retries.")
        return

    # 7. Save the working visualization script.
    code_path = "artifacts/data_visualization_code.py"
    try:
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Visualization code saved to {code_path}")
    except Exception as e:
        print(f"Error saving code: {e}")

    # ---- Phase 3: Report Generation ----
    print("\n========== Phase 3: Report Generation ==========")

    # 8. Invoke the report chat to generate the final report.
    logs = load_logs("agent_chat.log")
    logs_text = "\n".join(logs) if logs else "No logs available."

    report_prompt = (
        f"Generate a comprehensive data analysis report based on the following information.\n\n"
        f"Agent interaction logs:\n{logs_text}\n\n"
        f"Original data: {csv_data}\n\n"
        f"Analysis results (cleaned data and statistics):\n{analysis_result}\n\n"
        "The visualization has been saved to 'artifacts/data_visualization.png'. "
        "Reference it in the report as: ![Data Visualization](data_visualization.png)\n\n"
        "Follow the report template structure exactly."
    )

    await report_chat.add_chat_message(
        ChatMessageContent(role=AuthorRole.USER, content=report_prompt)
    )

    report_content = ""
    async for content in report_chat.invoke():
        log_agent_message(content)
        print(f"\n[{content.name}]:\n{content.content[:200]}...")
        if content.name == "ReportGenerator":
            report_content = content.content

    # 9. Save the final report.
    if report_content:
        save_final_report(report_content)
    else:
        print("Warning: No report content generated.")

    print("\n========== Workflow Complete ==========")
    print("Output files:")
    print(f"  - Cleaned data:        {cleaned_data_path}")
    print(f"  - Visualization code:  {code_path}")
    print(f"  - Visualization plot:  artifacts/data_visualization.png")
    print(f"  - Final report:        artifacts/final_report.md")


# -----------------
# Main Execution
# -----------------
if __name__ == "__main__":
    asyncio.run(main())