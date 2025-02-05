import PyPDF2 as pdf
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import google.generativeai as genai
import os
from langchain_groq import ChatGroq
import warnings
from langchain_groq import ChatGroq
from langchain.agents import AgentType, initialize_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool

warnings.filterwarnings("ignore")

## webpage configure 
st.set_page_config(page_title="DocuMed AI 🩺🔬", page_icon="🔬", layout="wide")
st.title("DocuMed AI 🩺🔬 — Your AI-Powered Medical Report Analyzer")

## Sidebar configuration
st.sidebar.title("Enter API Keys 🔑")
api_key = st.sidebar.text_input("Enter the GPT-4 API Key", type="password")  # Ganti jadi GPT-4 API Key
genai.configure(api_key=api_key)
os.environ['GPT4_API_KEY'] = api_key  # API Key untuk GPT-4

claude_api_key = st.sidebar.text_input("Enter the Claude API Key", type="password")  # Claude API Key
os.environ["CLAUDE_API_KEY"] = claude_api_key

serper_api_key = st.sidebar.text_input("Enter the Serper API Key", type="password")
os.environ["SERPER_API_KEY"] = serper_api_key

city = st.sidebar.selectbox("Select the city", ("Mumbai", "Pune", "Banglore"))

# Response function
def get_response(content, prompt, model_type="gpt4"):
    if model_type == "gpt4":
        model = genai.GenerativeModel("gpt-4-turbo")  # Gunakan GPT-4 Turbo
    elif model_type == "claude":
        model = genai.GenerativeModel("claude-3")  # Gunakan Claude 3
    response = model.generate_content([content, prompt])
    return response.text

# PDF processing
def input_pdf_setup(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

# Prompt for medical analysis
prompt1 = """You are a world-class medical expert specializing in hematology, internal medicine, and general health diagnostics. Your role is to act as a professional doctor, analyzing and interpreting blood test reports with precision, empathy, and clarity. Your responses should provide a comprehensive, patient-centric interpretation of the blood report while ensuring the information is accurate, actionable, and understandable."""

prompt2 = """You are a highly knowledgeable and professional doctor specializing in medical diagnostics and patient care. Your role is to analyze blood test reports, identify alarming or abnormal parameters, and present the findings in a clear, organized manner. Additionally, provide first-hand remedies and general advice for improving the flagged parameters."""

prompt3_temp = """show summary of alarming factors which can be easy to read, just a small paragraph that highlights the alarming factors in blood report"""

prompt3 = """Name top 5 best doctors along with hospital names in {city} based on other patients reviews, qualifications, experience, or similar symptoms using the context and city. Make sure you only recommend the doctors that are relevant to the patient's alarming symptoms mentioned in the context: {context}"""

def call_agent(context, city):
    search = GoogleSerperAPIWrapper()
    tools = [
        Tool(
            name='Intermediate Answer',
            func=search.run,
            description="useful for when you need to ask with search",
        )
    ]
    self_ask_agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        handle_parsing_errors=True
    )
    final_prompt = prompt3.format(context=context, city=city)
    result = self_ask_agent.run(final_prompt)
    return result

uploaded_file = st.file_uploader("Upload your blood report", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully")

# Interaction buttons
submit1 = st.button("How is my report?")
submit2 = st.button("Show me summary")
submit3 = st.button("Suggest some good doctors")

st.subheader("Response :")
if submit1:
    with st.spinner("Analyzing the report..."):
        text = input_pdf_setup(uploaded_file)
        response = get_response(text, prompt1, model_type="gpt4")  # Pilih model GPT-4
        st.write(response)
elif submit2:
    with st.spinner("Summarizing the report..."):
        text = input_pdf_setup(uploaded_file)
        response = get_response(text, prompt2, model_type="claude")  # Pilih model Claude
        st.write(response)
elif submit3:
    with st.spinner("Fetching doctors..."):
        text = input_pdf_setup(uploaded_file)
        response = get_response(text, prompt3_temp, model_type="gpt4")  # Pilih model GPT-4 untuk dokter
        result = call_agent(response, city)
        st.write(result)

# Footer section with custom styles
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #2c3e50;
            color: white;
            text-align: center;
            padding: 15px 0;
            font-size: 16px;
            z-index: 100;
            font-family: 'Arial', sans-serif;
        }
        .footer a {
            color: white;
            text-decoration: none;
        }
        .footer img {
            width: 25px;
            vertical-align: middle;
            margin-left: 10px;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
            border: none;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
    <div class="footer">
        <strong>Copyright © 2025 Farrel0xx</strong>
        <a href="https://github.com/Farrel0xx" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
