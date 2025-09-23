import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import os
from resume_parser import ResumeParser
from jd_parser import JDParser
from relevance_analyzer import RelevanceAnalyzer
from database import DatabaseManager
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Resume Relevance Check System | Innomatics Research Labs",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    db_manager = DatabaseManager()
    resume_parser = ResumeParser()
    jd_parser = JDParser()
    relevance_analyzer = RelevanceAnalyzer()
    return db_manager, resume_parser, jd_parser, relevance_analyzer

db_manager, resume_parser, jd_parser, relevance_analyzer = initialize_components()

# Enhanced CSS for professional and flashy design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Custom Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Navigation Cards */
    .nav-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: none;
    }
    
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .nav-card-hr {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .nav-card-applicant {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .nav-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .nav-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .nav-desc {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Score Cards */
    .score-high { 
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
    }
    
    .score-medium { 
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
    }
    
    .score-low { 
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
    }
    
    /* Form Styling */
    .stForm {
        background: rgba(255,255,255,0.05);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    /* Success/Error Styling */
    .stSuccess {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        color: white;
        border-radius: 10px;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Check if user type is selected
    if 'user_type' not in st.session_state:
        st.session_state.user_type = None
    
    if 'page' not in st.session_state:
        st.session_state.page = None
    
    # Main header
    st.markdown("""
    <div class="main-header animate-fade-in">
        <div class="main-title">🎯 Resume Relevance Check System</div>
        <div class="main-subtitle">Innomatics Research Labs | AI-Powered Recruitment Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    # User type selection
    if st.session_state.user_type is None:
        show_user_selection()
    elif st.session_state.user_type == "HR":
        show_hr_interface()
    elif st.session_state.user_type == "Applicant":
        show_applicant_interface()

def show_user_selection():
    st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 👥 Select Your Access Level")
        st.markdown("---")
        
        # HR Card
        if st.button("", key="hr_btn", help="Access HR Dashboard"):
            st.session_state.user_type = "HR"
            st.session_state.page = "Dashboard"
            st.rerun()
        
        st.markdown("""
        <div class="nav-card nav-card-hr" onclick="document.querySelector('[data-testid=\"baseButton-secondary\"]').click()">
            <div class="nav-icon">👨‍💼</div>
            <div class="nav-title">HR Team Access</div>
            <div class="nav-desc">Manage job descriptions, view analytics, and review candidates</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Applicant Card
        if st.button("", key="applicant_btn", help="Submit Resume"):
            st.session_state.user_type = "Applicant"
            st.rerun()
        
        st.markdown("""
        <div class="nav-card nav-card-applicant">
            <div class="nav-icon">🎓</div>
            <div class="nav-title">Job Applicant</div>
            <div class="nav-desc">Upload your resume and get instant relevance analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_hr_interface():
    # Sidebar navigation for HR
    st.sidebar.markdown("### 👨‍💼 HR Dashboard")
    st.sidebar.markdown("---")
    
    # Individual navigation buttons
    if st.sidebar.button("📊 Dashboard", width="stretch"):
        st.session_state.page = "Dashboard"
        st.rerun()
    
    if st.sidebar.button("📝 Upload Job Description",  width="stretch"):
        st.session_state.page = "Upload Job Description"
        st.rerun()
    
    if st.sidebar.button("📈 View Results",  width="stretch"):
        st.session_state.page = "View Results"
        st.rerun()
    
    if st.sidebar.button("📊 Analytics",  width="stretch"):
        st.session_state.page = "Analytics"
        st.rerun()
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Switch User Type",  width="stretch"):
        st.session_state.user_type = None
        st.session_state.page = None
        st.rerun()
    
    # Page content
    if st.session_state.page == "Dashboard":
        show_hr_dashboard()
    elif st.session_state.page == "Upload Job Description":
        upload_job_description()
    elif st.session_state.page == "View Results":
        view_results()
    elif st.session_state.page == "Analytics":
        show_analytics()
    else:
        show_hr_dashboard()

def show_applicant_interface():
    # Sidebar for applicants
    st.sidebar.markdown("### 🎓 Applicant Portal")
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tips for Success:**\n\n• Upload your latest resume\n• Choose the most relevant job\n• Review feedback carefully\n• Improve based on suggestions")
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Switch to HR Access", use_container_width=True):
        st.session_state.user_type = None
        st.session_state.page = None
        st.rerun()
    
    upload_resume()

def show_hr_dashboard():
    st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
    
    st.markdown("## 📊 HR Dashboard")
    
    # Get statistics
    stats = db_manager.get_dashboard_stats()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{stats['total_jds']}</div>
            <div class="metric-label">Active Jobs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{stats['total_resumes']}</div>
            <div class="metric-label">Total Candidates</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{stats['total_evaluations']}</div>
            <div class="metric-label">Evaluations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{stats['weekly_applications']}</div>
            <div class="metric-label">This Week</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("### 📈 Recent Applications")
    recent_evaluations = db_manager.get_recent_evaluations(limit=10)
    if recent_evaluations:
        df = pd.DataFrame([row[:8] for row in recent_evaluations], columns=[
        'ID', 'Job Title', 'Candidate', 'Email', 'Score', 'Verdict', 'Date', 'Location'
    ])


        
        # Create interactive table
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    help="Relevance Score",
                    min_value=0,
                    max_value=100,
                ),
                "Verdict": st.column_config.TextColumn(
                    "Verdict",
                    help="Suitability Level"
                )
            }
        )
    else:
        st.info("🚀 No applications yet. Start by uploading job descriptions!")
    
    st.markdown("</div>", unsafe_allow_html=True)

def upload_job_description():
    st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
    
    st.markdown("## 📝 Upload Job Description")
    
    with st.form("jd_upload_form", clear_on_submit=True):
        st.markdown("### 📋 Job Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.text_input("🎯 Job Title *", placeholder="e.g., Senior Data Scientist")
            company_name = st.text_input("🏢 Company Name *", placeholder="e.g., TechCorp Solutions")
            location = st.selectbox("📍 Location *", [
                "Hyderabad", "Bangalore", "Pune", "Delhi NCR", "Mumbai", "Chennai", "Other"
            ])
            
        with col2:
            experience_level = st.selectbox("💼 Experience Level *", [
                "Fresher (0-1 years)", "Junior (1-3 years)", "Mid-level (3-5 years)", 
                "Senior (5-8 years)", "Lead (8+ years)"
            ])
            job_type = st.selectbox("⏰ Job Type *", [
                "Full-time", "Part-time", "Internship", "Contract", "Remote"
            ])
            priority = st.selectbox("🚨 Priority", ["High", "Medium", "Low"])
        
        st.markdown("### 📄 Job Description File")
        st.markdown("**Upload the complete job description file (Required)**")
        
        uploaded_file = st.file_uploader(
            "Choose JD File *", 
            type=['txt', 'pdf', 'docx'],
            help="Upload the complete job description document"
        )
        
        submitted = st.form_submit_button("🚀 Process & Save Job Description",width="stretch")
        
        if submitted:
            if not all([job_title, company_name, location, experience_level, job_type, uploaded_file]):
                st.error("🚨 Please fill all required fields and upload the JD file!")
                return
            
            with st.spinner("🔄 Processing job description... Please wait"):
                try:
                    # Extract text from uploaded file
                    if uploaded_file.type == "text/plain":
                        job_description = str(uploaded_file.read(), "utf-8")
                    else:
                        job_description = resume_parser.extract_text_from_file(uploaded_file)
                    
                    if not job_description:
                        st.error("❌ Could not extract text from the uploaded file. Please try a different format.")
                        return
                    
                    # Parse JD
                    parsed_jd = jd_parser.parse_job_description(job_description)
                    
                    # Save to database
                    jd_id = db_manager.save_job_description({
                        'job_title': job_title,
                        'company_name': company_name,
                        'location': location,
                        'experience_level': experience_level,
                        'job_type': job_type,
                        'priority': priority,
                        'raw_description': job_description,
                        'parsed_data': parsed_jd
                    })
                    
                    st.success(f"✅ Job description processed and saved successfully! ID: {jd_id}")
                    
                    # Display parsed information
                    st.markdown("### 🎯 Extracted Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔥 Required Skills:**")
                        required_skills = parsed_jd.get('required_skills', [])
                        if required_skills:
                            for skill in required_skills[:10]:
                                st.markdown(f"• {skill}")
                        else:
                            st.info("No specific required skills identified")
                    
                    with col2:
                        st.markdown("**⭐ Preferred Skills:**")
                        preferred_skills = parsed_jd.get('preferred_skills', [])
                        if preferred_skills:
                            for skill in preferred_skills[:10]:
                                st.markdown(f"• {skill}")
                        else:
                            st.info("No specific preferred skills identified")
                    
                    if parsed_jd.get('qualifications'):
                        st.markdown("**🎓 Qualifications:**")
                        for qual in parsed_jd['qualifications'][:5]:
                            st.markdown(f"• {qual}")
                    
                    # Experience requirements
                    exp_req = parsed_jd.get('experience_required', {})
                    if exp_req.get('min_years', 0) > 0:
                        st.markdown(f"**💼 Experience Required:** {exp_req['min_years']} years minimum")
                            
                except Exception as e:
                    st.error(f"❌ Error processing job description: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def upload_resume():
    st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
    
    st.markdown("## 🎓 Submit Your Resume")
    
    # Get available job descriptions
    job_descriptions = db_manager.get_active_job_descriptions()
    
    if not job_descriptions:
        st.warning("⚠️ No job openings available at the moment. Please check back later!")
        return
    
    with st.form("resume_upload_form", clear_on_submit=True):
        st.markdown("### 👤 Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            candidate_name = st.text_input("👤 Full Name *", placeholder="Enter your full name")
            candidate_email = st.text_input("📧 Email Address *", placeholder="your.email@example.com")
            candidate_phone = st.text_input("📱 Phone Number", placeholder="+91-XXXXXXXXXX")
            
        with col2:
            # Job selection with better formatting
            job_options = [f"{jd[1]} | {jd[2]} | {jd[3]} (ID: {jd[0]})" for jd in job_descriptions]
            selected_job = st.selectbox("🎯 Select Job Position *", job_options)
            
            experience_years = st.number_input("💼 Years of Experience", min_value=0, max_value=50, value=0)
            current_location = st.text_input("📍 Current Location", placeholder="City, State")
        
        st.markdown("### 📄 Resume Upload")
        
        # Resume upload
        uploaded_resume = st.file_uploader(
            "📎 Upload Your Resume *", 
            type=['pdf', 'docx'],
            help="Supported formats: PDF, DOCX (Max size: 5MB)"
        )
        
        submitted = st.form_submit_button("🚀 Analyze My Resume",width="stretch")
        
        if submitted:
            if not all([candidate_name, candidate_email, selected_job, uploaded_resume]):
                st.error("🚨 Please fill all required fields and upload your resume!")
                return
            
            # Extract job ID
            job_id = selected_job.split("(ID: ")[1].split(")")[0]
            
            with st.spinner("🔄 Analyzing your resume... This may take a moment"):
                try:
                    # Parse resume
                    resume_text = resume_parser.extract_text_from_file(uploaded_resume)
                    if not resume_text:
                        st.error("❌ Could not extract text from your resume. Please try a different format.")
                        return
                    
                    parsed_resume = resume_parser.parse_resume(resume_text)
                    
                    # Get job description
                    job_data = db_manager.get_job_description(job_id)
                    if not job_data:
                        st.error("❌ Job description not found. Please try again.")
                        return
                    
                    # Analyze relevance
                    analysis_result = relevance_analyzer.analyze_relevance(
                        parsed_resume, 
                        job_data['parsed_data'],
                        resume_text,
                        job_data['raw_description']
                    )
                    
                    # Save results
                    evaluation_id = db_manager.save_evaluation({
                        'job_id': job_id,
                        'candidate_name': candidate_name,
                        'candidate_email': candidate_email,
                        'candidate_phone': candidate_phone,
                        'experience_years': experience_years,
                        'current_location': current_location,
                        'resume_text': resume_text,
                        'parsed_resume': parsed_resume,
                        'analysis_result': analysis_result
                    })
                    
                    st.success(f"✅ Resume analyzed successfully! Your evaluation ID: {evaluation_id}")
                    
                    # Display results
                    display_analysis_results(analysis_result, job_data['job_title'])
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing resume: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_analysis_results(analysis_result, job_title):
    st.markdown("---")
    st.markdown(f"## 📊 Your Analysis Results")
    st.markdown(f"**Position:** {job_title}")
    
    # Score and verdict
    score = analysis_result['relevance_score']
    verdict = analysis_result['verdict']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Score with color coding
        if score >= 70:
            st.markdown(f'<div class="score-high">🎯 Relevance Score<br><span style="font-size: 2rem;">{score}/100</span></div>', unsafe_allow_html=True)
        elif score >= 50:
            st.markdown(f'<div class="score-medium">📊 Relevance Score<br><span style="font-size: 2rem;">{score}/100</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="score-low">📉 Relevance Score<br><span style="font-size: 2rem;">{score}/100</span></div>', unsafe_allow_html=True)
    
    with col2:
        verdict_class = "score-high" if verdict == "High" else "score-medium" if verdict == "Medium" else "score-low"
        st.markdown(f'<div class="{verdict_class}">⭐ Suitability<br><span style="font-size: 1.5rem;">{verdict}</span></div>', unsafe_allow_html=True)
    
    with col3:
        match_percentage = score
        if match_percentage >= 70:
            icon = "🎉"
        elif match_percentage >= 50:
            icon = "👍"
        else:
            icon = "💪"
        st.markdown(f'<div class="metric-card">{icon} Match Level<br><span style="font-size: 1.5rem;">{match_percentage}%</span></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Your Matching Strengths")
        matching_skills = analysis_result.get('matching_skills', [])
        if matching_skills:
            for skill in matching_skills[:15]:
                st.markdown(f"🔥 {skill}")
        else:
            st.info("💡 No direct skill matches found. Consider highlighting transferable skills!")
    
    with col2:
        st.markdown("### 🎯 Skills to Develop")
        missing_skills = analysis_result.get('missing_skills', [])
        if missing_skills:
            for skill in missing_skills[:15]:
                st.markdown(f"📚 {skill}")
        else:
            st.success("🌟 Great! You have all the required skills!")
    
    # Improvement suggestions
    if analysis_result.get('suggestions'):
        st.markdown("### 💡 Personalized Improvement Plan")
        suggestions = analysis_result['suggestions']
        for i, suggestion in enumerate(suggestions, 1):
            st.markdown(f"**{i}.** {suggestion}")
    
    # Overall recommendation
    st.markdown("### 🎯 Overall Assessment")
    if score >= 70:
        st.success(f"🌟 **Excellent Match!** You're a strong candidate for this position. Your profile aligns well with the requirements.")
    elif score >= 50:
        st.warning(f"👍 **Good Potential!** You have a solid foundation. Focus on developing the missing skills to become an ideal candidate.")
    else:
        st.info(f"💪 **Growth Opportunity!** This role could be a stretch goal. Consider upskilling in the key areas mentioned above.")

def view_results():
    st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
    
    st.markdown("## 📈 Candidate Results")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        job_filter = st.selectbox("🎯 Filter by Job", ["All Jobs"] + [jd[1] for jd in db_manager.get_active_job_descriptions()])
    
    with col2:
        verdict_filter = st.selectbox("⭐ Filter by Verdict", ["All", "High", "Medium", "Low"])
    
    with col3:
        min_score = st.slider("📊 Minimum Score", 0, 100, 0)
    
    with col4:
        sort_by = st.selectbox("🔄 Sort by", ["Score (High to Low)", "Date (Latest)", "Name (A-Z)"])
    
    # Get filtered results
    evaluations = db_manager.get_evaluations_filtered(job_filter, verdict_filter, min_score)
    
    if evaluations:
        df = pd.DataFrame([row[:8] for row in evaluations], columns=[
        'ID', 'Job Title', 'Candidate', 'Email', 'Score', 'Verdict', 'Date', 'Location'
        ])

        
        # Sort data
        if sort_by == "Score (High to Low)":
            df = df.sort_values('Score', ascending=False)
        elif sort_by == "Date (Latest)":
            df = df.sort_values('Date', ascending=False)
        elif sort_by == "Name (A-Z)":
            df = df.sort_values('Candidate')
        
        # Display results with better formatting
        st.markdown(f"### 📋 Found {len(df)} candidates")
        
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    help="Relevance Score",
                    min_value=0,
                    max_value=100,
                ),
                "Verdict": st.column_config.TextColumn(
                    "Verdict",
                    help="Suitability Level"
                ),
                "Email": st.column_config.TextColumn(
                    "Email",
                    help="Contact Email"
                )
            }
        )
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"candidate_evaluations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            width="stretch"
        )
        
        # Detailed view
        st.markdown("### 🔍 Detailed Analysis")
        if st.checkbox("Show detailed candidate analysis"):
            selected_id = st.selectbox("Select candidate evaluation", df['ID'].tolist(), format_func=lambda x: f"ID {x}: {df[df['ID']==x]['Candidate'].iloc[0]}")
            if selected_id:
                detailed_result = db_manager.get_detailed_evaluation(selected_id)
                if detailed_result:
                    st.json(detailed_result['analysis_result'])
    else:
        st.info("🔍 No candidates match the current filters. Try adjusting your search criteria.")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_analytics():
    st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
    
    st.markdown("## 📊 Analytics Dashboard")
    
    # Get analytics data
    analytics_data = db_manager.get_analytics_data()
    
    if not analytics_data['evaluations']:
        st.info("📈 No evaluation data available yet. Analytics will appear once candidates start applying!")
        return
    
    # Score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Score Distribution")
        scores = [eval[4] for eval in analytics_data['evaluations']]
        
        fig = px.histogram(
            x=scores, 
            nbins=20, 
            title="Candidate Score Distribution",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            xaxis_title="Relevance Score",
            yaxis_title="Number of Candidates",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig,width="stretch")
    
    with col2:
        st.markdown("### ⭐ Verdict Distribution")
        verdicts = [eval[5] for eval in analytics_data['evaluations']]
        verdict_counts = pd.Series(verdicts).value_counts()
        
        colors = ['#11998e', '#f7971e', '#fc466b']
        fig = px.pie(
            values=verdict_counts.values, 
            names=verdict_counts.index, 
            title="Candidate Suitability Levels",
            color_discrete_sequence=colors
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig,width="stretch")
    
    # Job-wise analytics
    st.markdown("### 🎯 Job-wise Performance Analysis")
    job_performance = {}
    for eval in analytics_data['evaluations']:
        job = eval[1]
        score = eval[4]
        if job not in job_performance:
            job_performance[job] = []
        job_performance[job].append(score)
    
    if job_performance:
        job_stats = {}
        for job, scores in job_performance.items():
            job_stats[job] = {
                'avg_score': sum(scores) / len(scores),
                'total_applications': len(scores),
                'high_quality': len([s for s in scores if s >= 70])
            }
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Average Score by Job', 'Applications per Job', 'High-Quality Candidates', 'Score Trends'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        jobs = list(job_stats.keys())
        avg_scores = [job_stats[job]['avg_score'] for job in jobs]
        applications = [job_stats[job]['total_applications'] for job in jobs]
        high_quality = [job_stats[job]['high_quality'] for job in jobs]
        
        # Add traces
        fig.add_trace(go.Bar(x=jobs, y=avg_scores, name="Avg Score", marker_color='#667eea'), row=1, col=1)
        fig.add_trace(go.Bar(x=jobs, y=applications, name="Applications", marker_color='#f093fb'), row=1, col=2)
        fig.add_trace(go.Bar(x=jobs, y=high_quality, name="High Quality", marker_color='#11998e'), row=2, col=1)
        
        # Score trend over time
        dates = [eval[6] for eval in analytics_data['evaluations']]
        scores_trend = [eval[4] for eval in analytics_data['evaluations']]
        fig.add_trace(go.Scatter(x=dates, y=scores_trend, mode='lines+markers', name="Score Trend", line_color='#f7971e'), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("### 📈 Key Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_candidates = len(analytics_data['evaluations'])
    avg_score = sum(eval[4] for eval in analytics_data['evaluations']) / total_candidates
    high_performers = len([eval for eval in analytics_data['evaluations'] if eval[4] >= 70])
    improvement_needed = len([eval for eval in analytics_data['evaluations'] if eval[4] < 50])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{avg_score:.1f}</div>
            <div class="metric-label">Average Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{high_performers}</div>
            <div class="metric-label">High Performers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{(high_performers/total_candidates*100):.1f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{improvement_needed}</div>
            <div class="metric-label">Need Improvement</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()