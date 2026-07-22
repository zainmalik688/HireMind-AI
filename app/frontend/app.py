import re
import json
import requests
import streamlit as st

# Configure wide page layout
st.set_page_config(
    page_title="HireMind AI", 
    page_icon="📄", 
    layout="wide"
)

# ----------------------------------------------------
# Global CSS Styling (Enhanced Fonts, Spacing, & Metrics)
# ----------------------------------------------------
st.markdown("""
    <style>
        /* Increase global container width and baseline typography */
        .main .block-container { 
            max-width: 1200px; 
            padding-top: 2rem; 
            padding-bottom: 3rem;
        }
        
        /* Font size & line height adjustments for effortless readability */
        html, body, [class*="css"] { 
            font-size: 18px !important; 
        }
        
        h1 { 
            font-size: 2.5rem !important; 
            font-weight: 700 !important; 
            color: #1E88E5 !important; 
            margin-bottom: 0.2rem !important;
        }
        
        h2 { 
            font-size: 1.8rem !important; 
            font-weight: 600 !important; 
            margin-top: 1.8rem !important; 
            margin-bottom: 0.8rem !important;
            color: #333333;
        }
        
        h3 { 
            font-size: 1.4rem !important; 
            font-weight: 600 !important; 
            margin-top: 1.2rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        p, li, div { 
            font-size: 1.12rem !important; 
            line-height: 1.8 !important; 
        }

        /* Metric cards styling */
        .stMetric { 
            background-color: #1E222A; 
            padding: 18px; 
            border-radius: 12px; 
            border: 1px solid #313A46; 
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
        }
        
        div[data-testid="stMetricValue"] { 
            font-size: 2.2rem !important; 
            font-weight: 700 !important; 
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 8px 8px 0px 0px;
            padding-top: 10px;
            padding-bottom: 10px;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.title("📄 HireMind AI")
st.subheader("ATS Resume Parser & Technical Recruiter Feedback")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

def format_action_items(items):
    """Formats strings or lists into clean, well-spaced bullet points."""
    if isinstance(items, list):
        return "\n\n".join([f"* **{item}**" if not item.startswith("*") else item for item in items])
    elif isinstance(items, str) and items.strip():
        # If the backend returned a single string like "1. Fix this. 2. Fix that."
        parts = [p.strip() for p in re.split(r'\d+\.\s*', items) if p.strip()]
        if parts:
            return "\n\n".join([f"* {p}" for p in parts])
        return f"* {items}"
    return "* No specific items listed."

if uploaded_file is not None:
    if st.button("Analyze Resume", type="primary", use_container_width=True):
        with st.spinner("Parsing file and generating feedback..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post("http://127.0.0.1:8000/analyze", files=files)
                
                if response.status_code == 200:
                    raw_analysis = response.json().get("analysis", "No response text received.")
                    
                    # Safe JSON parsing logic
                    if isinstance(raw_analysis, dict):
                        ai_dict = raw_analysis
                    elif isinstance(raw_analysis, str) and raw_analysis.strip().startswith("{"):
                        try:
                            ai_dict = json.loads(raw_analysis)
                        except json.JSONDecodeError:
                            ai_dict = None
                    else:
                        ai_dict = None

                    if ai_dict:
                        # DYNAMIC SCORE EXTRACTION (Check multiple JSON keys safely)
                        score_breakdown = ai_dict.get("score_breakdown", {})
                        raw_ats = (
                            ai_dict.get("ats_score") 
                            or score_breakdown.get("calculated_final_score")
                            or "7.0/10"
                        )
                        
                        ats_str = str(raw_ats)
                        score_val = ats_str.split("/")[0].strip() if "/" in ats_str else ats_str

                        # Action Plan formatting with extra spacing and breathing room
                        action_data = ai_dict.get("action_plan", {})
                        immediate_str = format_action_items(action_data.get("immediate_fixes", []))
                        short_term_str = format_action_items(action_data.get("short_term_upgrades", []))
                        checklist_str = format_action_items(action_data.get("presubmission_checklist", []))

                        # Reconstruct Markdown report seamlessly from JSON structure
                        analysis_text = f"""# 📊 HireMind AI: Executive Analysis

- **Candidate Name**: {ai_dict.get('candidate_name', 'Candidate')}
- **Estimated ATS Score**: {ats_str}
- **Target Profiles**: {', '.join(ai_dict.get('target_profiles', ['AI/ML Engineer']))}

### Executive Summary
{ai_dict.get('executive_summary', '')}

---

## ✅ Verified Strengths & Working Elements

""" + "\n\n".join([f"* {s}" for s in ai_dict.get("strengths", [])]) + """

---

## 🚨 Critical Blockers & Evidence Deficits (Fix First)

""" + "\n\n".join([f"* {b}" for b in ai_dict.get("critical_blockers", [])]) + """

---

## 🔍 Keyword & Technical Skill Alignment

* **Strong Stack Matches**: {', '.join(ai_dict.get('skills_gap_analysis', {}).get('strong_matches', []))}

* **Missing High-Demand Skills**: {', '.join(ai_dict.get('skills_gap_analysis', {}).get('missing_high_demand', []))}

* **Layout & Section Tuning**: {str(ai_dict.get('skills_gap_analysis', {}).get('layout_tuning', ''))}

---

## 📝 STAR Method Bullet Point Rewrites (Adding Proof)

""" + "\n\n---\n\n".join([
    f"* **Original**: \"{r.get('original','')}\"\n\n  * **Optimized**: {r.get('optimized','')}\n\n  * **Why it works**: {r.get('why_it_works','')}" 
    for r in ai_dict.get("star_rewrites", []) if isinstance(r, dict)
]) + f"""

---

## 🎯 Priority Action Plan

### ⚡ Immediate Fixes (Next 30 Mins)
{immediate_str}

<br>

### 🚀 Short-Term Upgrades (1-2 Days)
{short_term_str}

<br>

### ✅ Pre-Submission Checklist
{checklist_str}
"""
                    else:
                        analysis_text = str(raw_analysis)
                        score_val = "N/A"

                    st.success("Analysis Complete!")
                    st.markdown("---")
                    
                    # Clean numeric conversion for metrics badge
                    try:
                        numeric_score = float(score_val)
                    except ValueError:
                        numeric_score = None

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if numeric_score is not None:
                            delta_msg = "Pass / Strong" if numeric_score >= 8.0 else ("Moderate / Optimization Recommended" if numeric_score >= 6.0 else "Action Required")
                            st.metric(
                                label="Estimated ATS Score", 
                                value=f"{numeric_score:.1f} / 10", 
                                delta=delta_msg
                            )
                        else:
                            st.metric(label="Estimated ATS Score", value="N/A")
                            
                    with col2:
                        target_list = ai_dict.get("target_profiles", ["AI/ML Engineer"]) if ai_dict else ["AI/ML Engineer"]
                        st.metric(label="Target Profile Match", value=target_list[0] if target_list else "Software Engineer")
                    with col3:
                        st.metric(label="System Status", value="Review Ready", delta_color="normal")

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Tabbed Section Layout
                    tab_overview, tab_analysis, tab_action = st.tabs([
                        "📊 Executive Overview & Strengths", 
                        "🔍 Detailed Gap Analysis", 
                        "🎯 Step-by-Step Action Plan"
                    ])

                    sections = analysis_text.split("---")

                    with tab_overview:
                        for sec in sections:
                            if any(k in sec for k in ["Executive Analysis", "Strengths"]):
                                st.markdown(sec)

                    with tab_analysis:
                        for sec in sections:
                            if any(k in sec for k in ["Critical Blockers", "Keyword & Technical", "STAR Method"]):
                                st.markdown(sec)

                    with tab_action:
                        for sec in sections:
                            if "Priority Action Plan" in sec:
                                st.markdown(sec)

                    st.markdown("---")
                    
                    # Download Report Button
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