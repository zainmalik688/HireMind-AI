import streamlit as st
import requests

st.set_page_config(
    page_title="HireMind AI - Resume Analytics",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🧠 HireMind AI")
st.subheader("Production-Grade Resume Parser & Analytics")
st.markdown("Upload your resume in PDF format to receive professional extraction analytics and matching insights.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf"], accept_multiple_files=False)

# Disable button if no file is selected
analyze_button = st.button(
    "Analyze Resume", 
    disabled=(uploaded_file is None),
    type="primary"
)

if analyze_button and uploaded_file:
    # Prepare payload
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    
    # Display modern loading status
    with st.spinner("Processing document layout and running deep analytics... Please wait."):
        try:
            # Connect to FastAPI backend
            response = requests.post("http://127.0.0.1:8000/api/analyze", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Successfully evaluated: {result.get('filename')}")
                
                st.markdown("### 📋 Analysis Results")
                st.markdown(result.get("analysis"))
            else:
                # Catch structured backend exceptions safely
                try:
                    error_detail = response.json().get("detail", "Unknown server application error.")
                except Exception:
                    error_detail = response.text
                st.error(f"Backend Error ({response.status_code}): {error_detail}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API service. Please verify Uvicorn is running on port 8000.")
        except Exception as e:
            st.error(f"An unexpected frontend execution exception occurred: {str(e)}")

elif analyze_button and not uploaded_file:
    st.warning("Please upload a valid PDF document before running execution analytics.")