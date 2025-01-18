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
from langchain_openai import OpenAI

warnings.filterwarnings("ignore")

## webpage configure 
st.set_page_config(page_title="PathoPro" , page_icon="🔬")
st.header("PathoPro 🩺🔬 — Your AI-Powered Medical Report Analyzer")


## Gemini Setup
st.sidebar.title="Enter the google API Key"
api_key=st.sidebar.text_input("Enter the Gemini API Key" , type="password")
genai.configure(api_key=api_key)
os.environ['GOOGLE_API_KEY']=api_key

## Llamma model inference
groq_api_key=st.sidebar.text_input("Enter the Groq API Key" , type="password")
os.environ["GROQ_API_KEY"]=groq_api_key
llm=ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile"
)

##Serper API Key :
serper_api_key=st.sidebar.text_input("Enter the Serper API Key" , type="password")
os.environ["SERPER_API_KEY"]=serper_api_key

city = st.sidebar.selectbox(
    "Select the city" , 
    ("Mumbai" , "Pune" , "Banglore")
    )

def get_response(content  , prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([content , prompt])
    return response.text


def input_pdf_setup(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text+=str(page.extract_text())
    return text



prompt1="""
You are a world-class medical expert specializing in hematology, internal medicine, and general health 
diagnostics. Your role is to act as a professional doctor, analyzing and interpreting blood test reports 
with precision, empathy, and clarity. Your responses should provide a comprehensive, patient-centric 
interpretation of the blood report while ensuring the information is accurate, actionable, and 
understandable.
If any file other than the medical report is given as input then just say wrong file uploaded please upload
the correct file
"""
# 8080808405
prompt2="""
You are a highly knowledgeable and professional doctor specializing in medical diagnostics and patient care.
Your role is to analyze blood test reports, identify alarming or abnormal parameters, and present the
findings in a clear, organized manner. Additionally, provide first-hand remedies and general advice 
for improving the flagged parameters.
Just show the summary in tabular format so the patient easily understand the alarming factors in the 
report .
Aprat from table
At the end show summary of alarming factors.which can be easy to read , just a small paragraph that highlights
the alarming factors in blood report
"""

prompt3_temp="""
show summary of alarming factors.which can be easy to read , just a small paragraph that highlights
the alarming factors in blood report
"""

prompt3="""
Name top 5 best doctors along with hospital names in {city}
based on  other patients reviews, qualifications, experience ho suffered similar symptoms using the 
context and city : 
{context}
Suggest the doctors in list format , so that they are easy to read
If there are no specialized doctors in the {city} then recommend the doctors along the hospitals from
the city near to {city}
Make sure u only recommend the doctors that are relevenat to the patients mentioned in the {context}
Plz mention hospitals
Make sure the response is personalized for the user.
"""

def call_agent(context , city):
    search=GoogleSerperAPIWrapper()
    tools=[
        Tool(
            name = 'Intermediate Answer',
            func=search.run,
            description="useful for when you need to ask with search",
        )
    ]


    self_ask_agent=initialize_agent(
        tools , 
        llm , 
        agent="zero-shot-react-description" , 
        handle_parsing_errors=True
    )
    
    final_prompt=prompt3.format(context=context , city=city)

    result=self_ask_agent.run(final_prompt)
    
    return result
    
uploaded_file=st.file_uploader("Upload your blood report" , type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully")

submit1=st.button("How is my report?")

submit2=st.button("Show me summary")

submit3=st.button("Suggest some good doctors")

st.subheader("Response : ")
if submit1:
    text=input_pdf_setup(uploaded_file)
    response=get_response(text , prompt1)
    st.write(response)


elif submit2:
    text=input_pdf_setup(uploaded_file)
    response=get_response(text , prompt2)
    st.write(response)

elif submit3:
    text=input_pdf_setup(uploaded_file)
    response=get_response(text , prompt3_temp)
    result=call_agent(response , city)
    st.write(result)
    
    
    
    
    

    
