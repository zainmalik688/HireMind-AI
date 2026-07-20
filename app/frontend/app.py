import re
import streamlit as st
import requests

# Set wide layout so metric cards and tabs have plenty of horizontal space
st.set_page_config(
    page_title="HireMind AI", 
    page_icon="📄", 
    layout="wide"
)

st.title("📄 HireMind AI")
st.subheader("ATS Resume Parser & Technical Recruiter Feedback")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    if st.button("Analyze Resume", type="primary", use_container_width=True):
        with st.spinner("Parsing file and generating feedback..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post("http://127.0.0.1:8000/analyze", files=files)
                
                if response.status_code == 200:
                    analysis_text = response.json().get("analysis", "No response text received.")
                    
                    st.success("Analysis Complete!")
                    st.markdown("---")
                    
                    # ----------------------------------------------------
                    # 1. Custom Metric Badges (Top-level SaaS Overview)
                    # ----------------------------------------------------
                    # Extract estimated ATS Score using regex if available
                    score_match = re.search(r"Estimated ATS Score\*\*:\s*([0-9\.]+)(?:/10)?", analysis_text)
                    ats_score = score_match.group(1) if score_match else "N/A"
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label="Estimated ATS Score", 
                            value=f"{ats_score} / 10" if ats_score != "N/A" else "N/A", 
                            delta="-5.5 pts" if ats_score != "N/A" and float(ats_score) < 7 else "Pass"
                        )
                    with col2:
                        st.metric(label="Target Profile Match", value="AI/ML & Vision")
                    with col3:
                        st.metric(label="System Status", value="Action Required", delta_color="inverse")

                    st.markdown("---")

                    # ----------------------------------------------------
                    # 2. Tabbed Section Layout
                    # ----------------------------------------------------
                    tab_overview, tab_analysis, tab_action = st.tabs([
                        "📊 Executive Overview & Strengths", 
                        "🔍 Detailed Gap Analysis", 
                        "🎯 Step-by-Step Action Plan"
                    ])

                    # Split report using backend Markdown horizontal rules
                    sections = analysis_text.split("---")

                    with tab_overview:
                        for sec in sections:
                            if "Executive Analysis" in sec or "Strengths" in sec:
                                st.markdown(sec)

                    with tab_analysis:
                        for sec in sections:
                            if any(k in sec for k in ["Critical Blockers", "Gap Analysis", "Bullet Point Optimization"]):
                                st.markdown(sec)

                    with tab_action:
                        for sec in sections:
                            if "Action Plan" in sec:
                                st.markdown(sec)

                    st.markdown("---")
                    
                    # ----------------------------------------------------
                    # 3. Download Report Button
                    # ----------------------------------------------------
                    st.download_button(
                        label="📥 Download Full Analysis Report",
                        data=analysis_text,
                        file_name="HireMind_AI_Resume_Analysis.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                else:
                    st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error occurred.')}")
            except Exception as e:
                st.error(f"Could not connect to backend server: {str(e)}")