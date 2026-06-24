import tempfile
import csv
import streamlit as st
import pandas as pd
from agno.agent import Agent
import plotly.express as px
from reportlab.pdfgen import canvas
#from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.tools.duckdb import DuckDbTools
from agno.tools.pandas import PandasTools

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            try:
                df = pd.read_csv(
                file,
                encoding='utf-8',
                na_values=['NA', 'N/A', 'missing']
            )
            except UnicodeDecodeError:
                file.seek(0)
                df = pd.read_csv(
                    file,
                    encoding='latin1',
                    na_values=['NA', 'N/A', 'missing']
                )
            #df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep as is if conversion fails
                    pass
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # Save the DataFrame to the temporary CSV file with quotes around string fields
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Streamlit app
st.title("📊 Data Analyst Agent")
if "messages" not in st.session_state:
    st.session_state.messages = []
# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    #openai_key = st.text_input("OpenAI_API_Key", type="password")
    openrouter_key = st.text_input(
    "OpenRouter API Key",
    type="password"
)

if openrouter_key:
    st.session_state.openrouter_key = openrouter_key
    st.success("API key saved!")
else:
    st.warning("Please enter your OpenRouter API key to proceed.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and "openrouter_key" in st.session_state:
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rows",
            len(df)
        )

        col2.metric(
            "Columns",
            len(df.columns)
       )

        col3.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )
        st.write("Uploaded Data:")
        st.dataframe(df)  # Use st.dataframe for an interactive table
        
        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)

        ## Quick visualization
        st.subheader("📊 Dataset Visualization")

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) > 0:
            selected_col = st.selectbox(
            "Select a column to visualize",
            numeric_cols
            )

            chart_type = st.selectbox(
                "Select Chart Type",
                ["Histogram", "Box Plot","Bar Chart","Scatter Plot","Pie Chart"]
            )

            if chart_type == "Histogram":
                fig = px.histogram(
                    df,
                    x=selected_col,
                    title=f"Distribution of {selected_col}"
                )
            elif chart_type == "Box Plot":

                fig = px.box(
                df,
                y=selected_col,
                title=f"Box Plot of {selected_col}"
            )

            elif chart_type == "Line Chart":

                fig = px.line(
                df,
                y=selected_col,
                title=f"Line Chart of {selected_col}"
            )

            elif chart_type == "Bar Chart":

                fig = px.bar(
                df.head(20),
                y=selected_col,
                title=f"Bar Chart of {selected_col}"
            )

            elif chart_type == "Scatter Plot":

                fig = px.scatter(
                df,
                x=df.index,
                y=selected_col,
                title=f"Scatter Plot of {selected_col}"
            )

            elif chart_type == "Pie Chart":

                value_counts = df[selected_col].value_counts().head(10)

                fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Pie Chart of {selected_col}"
            )
            
            else:
                fig = px.box(
                    df,
                    y=selected_col,
                    title=f"Box Plot of {selected_col}"
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("🤖 Quick Statistics")
            st.write(df[selected_col].describe())

        # Initialize DuckDbTools
        duckdb_tools = DuckDbTools()
        
        # Load the CSV file into DuckDB as a table
        duckdb_tools.load_local_csv_to_table(
            path=temp_path,
            table="uploaded_data",
        )
        
        # Initialize the Agent with DuckDB and Pandas tools
        data_analyst_agent = Agent(
            #model=OpenAIChat(id="gpt-4o", api_key=st.session_state.openai_key),
            model=OpenAIChat(
                id="deepseek/deepseek-chat",
                api_key=st.session_state.openrouter_key,
                base_url="https://openrouter.ai/api/v1"
            ),
            tools=[duckdb_tools, PandasTools()],
            system_message="You are an expert data analyst. Use the 'uploaded_data' table to answer user queries. Generate SQL queries using DuckDB tools to solve the user's questions. If the answer cannot be found in the data, state that clearly.",
            markdown=True,
        )
        
        # Initialize code storage in session state
        if "generated_code" not in st.session_state:
            st.session_state.generated_code = None

        
        # Main query input widget
        # Show previous chat history
        for msg in st.session_state.messages:
            st.write("🧑 You:", msg["question"])
            st.write("🤖 Analyst:", msg["answer"])
            st.divider()
        user_query = st.text_area("Ask a query about the data:")
        
        # Add info message about terminal output
        st.info("💡 Check your terminal for a clearer output of the agent's response")
        
        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('Processing your query...'):
                        # Get the response from the agent
                        response = data_analyst_agent.run(user_query)

                        # Extract the content from the response object
                        if hasattr(response, 'content'):
                            response_content = response.content
                        else:
                            response_content = str(response)

                    # Display the response in Streamlit
                    # Save chat history
                    st.session_state.messages.append({
                        "question": user_query,
                        "answer": response_content
                    })
                    st.markdown(response_content)
                    
                    st.download_button(
                        label="📄 Download Report",
                        data=response_content,
                        file_name="analysis_report.txt",
                        mime="text/plain"
                  )
                
                    
                except Exception as e:
                    st.error(f"Error generating response from the agent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")