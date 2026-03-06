In this project, you will complete a Python script that uses the Semantic Kernel SDK to orchestrate a team of AI agents. Your goal is to build a fully automated pipeline that can take raw CSV data and, through a series of specialized agents, clean it, analyze it, visualize it, and produce a final, comprehensive report.

You will be working with a starter script that contains several \<TODO\> sections. Your task is to complete these sections in the specified order to create a functional end-to-end workflow.

## **Part 1: Model Deployment and Setup**

Before writing any code, you must deploy the AI model on Microsoft Azure and configure your local environment to connect to it.

### **Step 1: Deploy Your Model on Azure**

First, you need to create an Azure AI Foundry resource and deploy a gpt-4.1 model.

1. **Log into the Microsoft Azure Portal** using the credentials provided to you. It's recommended to use an incognito or private browser window.  
2. From the Azure portal home page, click on **\+ Create a resource** and search the Marketplace for Azure AI Foundry.  
3. Select **Azure AI Foundry** from the results and click **Create**.  
4. On the configuration page, use your existing subscription and resource group. Provide a **unique name** for your new resource.  
5. Proceed through the "Network," "Identity," and "Tags" tabs. On the **Review \+ submit** tab, click **Create**.  
6. Once the deployment is complete (this may take several minutes), navigate to the newly created resource page.  
7. From your resource's overview page, click on **Go to Azure AI Foundry portal**. This will open the Azure AI Studio.  
8. In the left-hand navigation menu, go to **Models \+ endpoints**.  
9. Click **\+ Deploy model** and select **Deploy a base model**. From the model list, choose **gpt-4.1** and click **Confirm**.  
10. Review the configuration and click **Create and deploy**.  
11. Once the deployment succeeds, you will be directed to the endpoint's detail page. Here, you will find the **Endpoint URL** and the authentication **Key**. Keep these two values safe, as you will need them in the next step.

### **Step 2: Configure Your Local Environment**

Now, connect your Python script to the model you just deployed.

1. In your project's root directory, create a new file named .env.  
2. Add the following two lines to this file, replacing the placeholder text with the credentials you obtained from the Azure portal:  
   AZURE_OPENAI_KEY=YOUR_KEY_HERE  
   URL=https://YOUR-ENDPOINT-URL-HERE/
3. Ensure that load_dotenv() is called at the beginning of your Python script to load these secrets into your environment.

## **Part 2: Coding the Agentic Workflow**

With your environment configured, it's time to complete the Python script. Follow these steps in order.

### **Step 3: Imports and Kernel Initialization**

First, ensure all necessary libraries are imported and that the Semantic Kernel is properly initialized.

1. **Complete Imports:** Locate the \<TODO: IMPORTS\> block at the top of the file and make sure all required components from os, asyncio, pandas, json, logging, dotenv, and semantic_kernel are imported.  
2. **Initialize Kernel:** Find the \<TODO: KERNEL\> block. Here, you must:  

- Initialize the Kernel.  
- Define the AzureChatCompletion service, passing your API_KEY, BASE_URL, and an API_VERSION of "2024-05-01-preview".  
- Add the chat service to the kernel instance.

### **Step 4: Implement Supporting Logic (Helper Functions)**

Next, complete the helper functions that handle file I/O. These are crucial for the agents to receive data and save their work.

- Locate the \<TODO\> markers within the helper functions and complete their logic. This includes functions for loading the user-selected CSV data, reading agent interaction logs, and saving the final report and other artifacts.

### **Step 5: Build the Agents and Teams**

Now, you will create the individual AI agents and assemble them into their collaborative groups.

1. **Create the Agent Factory:** In the \<TODO: create_agent()\> block, complete the function to return a ChatCompletionAgent instance. This factory will be used to create all your agents.  
2. **Define All Agents:** In the \<TODO: AGENTS\> block, you will write the instruction prompts for each agent and then instantiate them using the factory. Each agent needs a name, your custom prompt, and a specific temperature setting. Use the table below as a guide to write a precise and effective prompt for each agent. The goal is to control the agent's behavior and ensure its output is in the exact format the workflow expects.

|   Agent Name   |   Core Responsibility   |   Key Instructions for Your Prompt   |   Temperature   |
| --- | --- | --- | --- |
|   PythonExecutorAgent   |   Generates runnable Python code for visualization.   |   \- Must use matplotlib to plot original (blue) and cleaned (green) data on a single line chart. \- Must save the plot to artifacts/data_visualization.png. \- Must ensure the artifacts directory exists. \- The output must be only the raw Python code, with no explanations or comments.   |   0.1   |
|   DataCleaning   |   Cleans the raw dataset.   |   \- Must act as a "Data Cleaning Assistant." \- It should date the data and remove outliers. \- Before cleaning, it must present its cleaning plan. \- The final output must be only the cleaned data.   |   0.7   |
|   DataStatistics   |   Calculates descriptive statistics.   |   \- Must act as a "Data Statistics Assistant." \- When given data, its output must be only the statistical description, with no commentary.   |   0.5   |
|   AnalysisChecker   |   Validates the analysis phase.   |   \- Must act as a "Data Validation Auditor." \- It needs to verify that outliers were removed before statistics were calculated. \- Instruct it to use the rules provided in the Data_Quality_Instructions.txt file.   |   0.2   |
|   ReportGenerator   |   Writes the final markdown report.   |   \- Must act as a "Report Generator." \- It should synthesize the cleaning, statistics, and validation steps into one report. \- Instruct it to follow the structure defined in the Report_Instructions.txt file.   |   1.0   |
|   ReportChecker   |   Validates the final report.   |   \- Must act as a "Report Validation Auditor." \- It needs to check if the report is complete and accurate. \- Instruct it to validate the report against the rules in Report_Instructions.txt.   |   0.2   |

3. **Create Group Chats:** In the \<TODO: GROUP CHATS\> block, create the three AgentGroupChat instances:  

- analysis_chat: Contains the DataCleaning, DataStatistics, and AnalysisChecker agents.  
- code_chat: Contains only the PythonExecutorAgent.  
- report_chat: Contains the ReportGenerator and ReportChecker agents.

### **Step 6: Orchestrate the Main Workflow**

This is the final and most critical coding step. In the \<TODO: MAIN\> block, you will implement the logic that runs the entire pipeline from start to finish. Ensure your workflow performs these steps in the correct sequence and uses the exact prompts specified.

**Workflow Sequence:**

1. Load the user-selected CSV data and format it into an initial prompt.  
2. Invoke the analysis_chat to perform data cleaning and statistical analysis.  
3. After the analysis is complete, prompt the user for **human approval** to continue. The workflow should only proceed if the user enters "yes".  
4. Save the cleaned data from the analysis chat to a JSON file.  
5. Pass the cleaned data to the code_chat to generate Python visualization code.  
6. Execute the generated code in a **retry loop**. If the code fails, capture the error and send it back to the code_chat to ask for a fix.  
7. Once the code executes successfully, save the working script and the generated plot to the artifacts/ directory.  
8. Load the full agent interaction log and pass it to the report_chat to generate the final summary.  
9. Save the final markdown report.

## **Part 3: Execution and Final Deliverables**

After completing all the \<TODO\> sections in the script, run the entire workflow. If your implementation is correct, the script will execute from start to finish and produce four output files in the specified locations.

Verify that the following files have been created:

|   File   |   Location   |   Description   |
| --- | --- | --- |
|   Cleaned Data   |   data-cleaned.json   |   The cleaned dataset in JSON format.   |
|   Working Visualization Code   |   artifacts/data_visualization_code.py   |   The final, successfully executed Python script.   |
|   Generated Plot   |   artifacts/data_visualization.png   |   The PNG image of the data visualization.   |
|   Final Markdown Report   |   artifacts/final_report.md   |   The complete, structured report of the entire process.   |

Congratulations\! You have successfully built and orchestrated a multi-agent AI workflow.