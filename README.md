To run the above Python code, you need to have the required dependencies installed and ensure all the necessary configurations are in place. Below is a step-by-step explanation on how to run this code:
________________________________________
1. Set Up Your Environment
Before running the code, ensure you have Python installed (preferably Python 3.9+). You also need to install the required libraries and APIs.
Install Required Libraries:
Run the following command in your terminal to install the dependencies:
pip install streamlit requests duckduckgo-search openai PyPDF2
Optional:
•	If you use a virtual environment, activate it before running the above command.
________________________________________
2. Replace API Keys
You need to replace placeholders like "your_api_key" with valid API keys:
•	OpenRouter API Key: Replace "your_api_key" with a valid API key from OpenRouter.
•	Tomorrow.io API Key: Replace "Your_api_key" with a valid API key from Tomorrow.io.
________________________________________
3. Save the Code
Save the code in a file with a .py extension, for example, multitool_chat_app.py.
________________________________________
4. Run the Streamlit Application
Run the Streamlit application from the terminal with the following command:
streamlit run multitool_chat_app.py
•	This command will start a local server and provide a URL, usually http://localhost:8501/.
________________________________________
5. Explore the Application
The application is interactive and has multiple features accessible via a sidebar menu. Here's what each feature does:
a. Query Processing
•	Functionality: Allows you to process queries like weather updates, web searches, or LLM completions.
•	Input: Type a query in the input box.
•	Outputs: 
o	Fetches weather (current or for a specific location).
o	Performs DuckDuckGo searches.
o	Opens URLs for image or picture explanation tools.
________________________________________
b. Weather Information
•	Functionality: Fetches weather for a specified location or your current location.
•	Steps: 
1.	Enter a location or leave the input blank for current location.
2.	Click "Get Weather."
3.	View the detailed weather report.
________________________________________
c. PDF Summarization
•	Functionality: Uploads a PDF file, displays its content, and allows you to ask queries about the document.
•	Steps: 
1.	Upload a PDF file.
2.	Enter queries in the text area (one query per line).
3.	Click "Submit Queries."
4.	View answers generated by the LLM.
________________________________________
d. Image Search & Picture Explanation
•	These features open external tools like Meta AI or Copilot in a browser.
________________________________________
e. History
•	Displays the history of all actions taken during the session, including user inputs and outputs.
________________________________________
6. Debugging & Logs
•	If the app fails to run, check the terminal for errors.
•	Ensure that all API keys are correctly set and that your internet connection is active.
________________________________________
7. Enhancing the App
You can improve the app by:
•	Adding error handling for missing or invalid API keys.
•	Storing history data across sessions using persistent storage (e.g., a database or file).
•	Improving the UI by using Streamlit widgets.
________________________________________
Example Usage
1.	Launch the app using streamlit run multitool_chat_app.py.
2.	Select "PDF Summarization" from the sidebar.
3.	Upload a PDF file and type queries like: 
o	"Summarize the document."
o	"What are the key findings?"
4.	Click "Submit Queries" to see the answers.
Let me know if you encounter any issues!
________________________________________
Requirement pip is
streamlit
requests
duckduckgo-search
openAi
PyPDF2



