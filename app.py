import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.graph_objects as go
from recommendation_engine import HealthRecommendationEngine

st.set_page_config(page_title="Disease & Medicine Prediction System", layout="centered")

st.markdown("""
    <style>
    .main {
        max-width: 1000px;
        padding: 1rem;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .severity-high { border-left-color: #f44336; }
    .severity-moderate { border-left-color: #ff9800; }
    .severity-low { border-left-color: #4caf50; }
    
    .recommendation-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.8rem 0;
        transition: transform 0.2s;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit default elements */
    .stDeployButton, .stApp > header, footer { 
        visibility: hidden;
    }
    
    /* Better spacing */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        border-radius: 8px !important;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0 12px;
            font-size: 0.9rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
    st.session_state.model_loaded = False
    st.session_state.selected_symptoms = {}
    st.session_state.report = None

@st.cache_resource
def load_engine():
    """Load the recommendation engine (cached)"""
    engine = HealthRecommendationEngine()
    
    if engine.load_model() and engine.load_recommendation_data():
        return engine
    return None

def display_header():
    """Display application header"""
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='color: #2c3e50; margin-bottom: 0.5rem;'>
                ❤️ Disease & Medicine Prediction System
            </h1>
            <p style='color: #6c757d; margin-top: 0;'>
                Simple and private symptom checker with AI-powered insights
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

def symptom_selection_interface(engine):
    """Create symptom selection interface"""
    st.sidebar.header("🔍 Select Symptoms")
    
    # Get all symptoms
    all_symptoms = sorted(engine.get_all_symptoms())
    
    # Search functionality with better styling
    search_term = st.sidebar.text_input(
        "Search symptoms...",
        placeholder="Start typing to search",
        help="Type to filter symptoms"
    )
    
    # Filter symptoms based on search
    filtered_symptoms = [s for s in all_symptoms 
                        if search_term.lower() in s.lower()] if search_term else all_symptoms
    
    # Display selected count
    selected_count = sum(1 for v in st.session_state.selected_symptoms.values() if v == 1)
    
    # Selected symptoms counter
    st.sidebar.markdown(
        f"""<div style='display: flex; justify-content: space-between; align-items: center; margin: 10px 0;'>
            <span style='font-weight: 600;'>Selected: {selected_count}</span>
            <button onclick='window.location.reload()' style='background: none; border: none; color: #4CAF50; cursor: pointer; font-size: 0.9em;'>
                Clear all
            </button>
        </div>""", 
        unsafe_allow_html=True
    )
    
    # Create scrollable container for symptoms
    with st.sidebar.container():
        selected_symptoms = {}
        
        # Display symptoms in a more compact way
        if filtered_symptoms:
            cols = st.columns(2)
            for i, symptom in enumerate(filtered_symptoms[:50]):  # Limit to 50 for performance
                with cols[i % 2]:
                    is_selected = st.checkbox(
                        symptom.replace('_', ' ').title(),
                        key=f"symptom_{symptom}",
                        value=st.session_state.selected_symptoms.get(symptom, False),
                        label_visibility="visible"
                    )
                    if is_selected:
                        selected_symptoms[symptom] = 1
                        st.session_state.selected_symptoms[symptom] = True
                    else:
                        st.session_state.selected_symptoms[symptom] = False
        
        if not filtered_symptoms:
            st.sidebar.info("No symptoms found. Try a different search term.")
        elif len(filtered_symptoms) > 50:
            st.sidebar.info(f"Showing 50 of {len(filtered_symptoms)} symptoms. Use search or 'View All' to see more.")
    
    # Add View All Symptoms button at the bottom
    st.sidebar.markdown("---")
    if st.sidebar.button("View All Symptoms", use_container_width=True, key="view_all_btn"):
        with st.sidebar.expander("All Available Symptoms", expanded=True):
            st.write("<div style='max-height: 400px; overflow-y: auto;'>", unsafe_allow_html=True)
            cols = st.columns(2)
            for i, symptom in enumerate(all_symptoms):
                with cols[i % 2]:
                    is_selected = st.checkbox(
                        symptom.replace('_', ' ').title(),
                        key=f"all_symptom_{symptom}",
                        value=st.session_state.selected_symptoms.get(symptom, False),
                        label_visibility="visible"
                    )
                    if is_selected:
                        st.session_state.selected_symptoms[symptom] = True
                        selected_symptoms[symptom] = 1
                    else:
                        st.session_state.selected_symptoms[symptom] = False
            st.write("</div>", unsafe_allow_html=True)
    
    return selected_symptoms

def display_prediction_results(report):
    """Display prediction results"""
    st.markdown("## 🎯 Diagnosis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class='prediction-box'>
                <h3 style='color: #2c3e50; margin-bottom: 10px;'>Predicted Disease</h3>
                <h2 style='color: #4CAF50; margin: 0;'>{report['prediction']['disease']}</h2>
                <p style='color: #7f8c8d; margin-top: 10px;'>
                    Confidence: {report['prediction']['confidence']:.1%}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        severity_class = f"severity-{report['severity']['level'].lower()}"
        st.markdown(f"""
            <div class='prediction-box {severity_class}'>
                <h3 style='color: #2c3e50; margin-bottom: 10px;'>Severity Level</h3>
                <h2 style='margin: 0;'>{report['severity']['level']}</h2>
                <p style='color: #7f8c8d; margin-top: 10px;'>
                    Score: {report['severity']['score']}/7
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='prediction-box'>
                <h3 style='color: #2c3e50; margin-bottom: 10px;'>Symptoms Count</h3>
                <h2 style='color: #3498db; margin: 0;'>{report['symptoms']['count']}</h2>
                <p style='color: #7f8c8d; margin-top: 10px;'>
                    Active symptoms detected
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Confidence chart
    st.markdown("### 📊 Top 3 Possible Diagnoses")
    
    top_3 = report['prediction']['top_3_predictions']
    diseases = [pred['disease'] for pred in top_3]
    probabilities = [pred['probability'] * 100 for pred in top_3]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities,
            y=diseases,
            orientation='h',
            marker=dict(
                color=['#4CAF50', '#2196F3', '#FFC107'],
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f"{p:.1f}%" for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence Comparison",
        xaxis_title="Confidence (%)",
        yaxis_title="Disease",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_disease_description(report):
    """Display disease description"""
    st.markdown("## 📖 Disease Information")
    
    st.markdown(f"""
        <div class='recommendation-card'>
            <h3 style='color: #2c3e50;'>{report['prediction']['disease']}</h3>
            <p style='font-size: 16px; line-height: 1.6; color: #34495e;'>
                {report['description']}
            </p>
        </div>
    """, unsafe_allow_html=True)

def display_recommendations(report):
    """Display all recommendations"""
    st.markdown("## 💡 Personalized Recommendations")
    
    # Create tabs for different recommendations
    tab1, tab2, tab3, tab4 = st.tabs([
        "💊 Medications", 
        "🥗 Diet Plan", 
        "🏃 Lifestyle & Exercise", 
        "⚕️ Precautions"
    ])
    
    with tab1:
        st.markdown("### Recommended Medications")
        if report['medications']:
            for i, med in enumerate(report['medications'], 1):
                st.markdown(f"""
                    <div class='recommendation-card'>
                        <strong>{i}. {med}</strong>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific medication recommendations available.")
        
        st.warning("⚠️ **Important:** Always consult with a healthcare professional before taking any medication.")
    
    with tab2:
        st.markdown("### Dietary Recommendations")
        if report['diet']:
            col1, col2 = st.columns(2)
            for i, diet in enumerate(report['diet']):
                with col1 if i % 2 == 0 else col2:
                    st.markdown(f"""
                        <div class='recommendation-card'>
                            ✓ {diet}
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No specific diet recommendations available.")
    
    with tab3:
        st.markdown("### Lifestyle & Exercise Recommendations")
        if report['workouts']:
            for i, workout in enumerate(report['workouts'], 1):
                st.markdown(f"{i}. {workout}")
        else:
            st.info("No specific workout recommendations available.")
    
    with tab4:
        st.markdown("### Important Precautions")
        if report['precautions']:
            for i, precaution in enumerate(report['precautions'], 1):
                st.markdown(f"""
                    <div class='recommendation-card' style='background-color: #fff3cd; border-left: 5px solid #ffc107;'>
                        <strong>{i}. {precaution}</strong>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific precautions available.")

def display_active_symptoms(report):
    """Display active symptoms"""
    symptoms = report['symptoms']['active_symptoms']
    
    # Display in a clean, card-like format
    st.markdown("### Active Symptoms")
    cols = st.columns(3)
    for i, symptom in enumerate(symptoms):
        with cols[i % 3]:
            st.markdown(
                f"<div style='background: #f8f9fa; padding: 0.5rem 1rem; "
                f"border-radius: 8px; margin: 0.2rem 0; font-size: 0.9rem;'>"
                f"• {symptom.replace('_', ' ').title()}</div>",
                unsafe_allow_html=True
            )

def main():
    """Main application function"""
    display_header()
    
    # Initialize session state for engine
    if 'engine' not in st.session_state:
        st.session_state.engine = None
        st.session_state.model_loaded = False
        st.session_state.selected_symptoms = {}
        st.session_state.report = None
    
    # Load engine
    if st.session_state.engine is None:
        with st.spinner("Loading AI model..."):
            st.session_state.engine = load_engine()
            
            if st.session_state.engine is None:
                st.error("Failed to load the model. Please ensure the model is trained.")
                st.info("Run `python model_training.py` to train the model first.")
                st.stop()
            else:
                st.session_state.model_loaded = True
                st.session_state.engine = st.session_state.engine
    
    engine = st.session_state.engine
    
    # Symptom selection
    selected_symptoms = symptom_selection_interface(engine)
    
    # Main content area
    if not selected_symptoms:
        # Welcome screen
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0 2rem;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>💊</div>
                <h2 style='color: #2c3e50;'>How are you feeling today?</h2>
                <p style='color: #6c757d; max-width: 600px; margin: 0 auto 2rem;'>
                    Select your symptoms from the sidebar to get started with a quick health assessment.
                    Your information is private and never stored.
                </p>
            </div>
            
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin: 2rem 0;'>
                <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 1.8rem; margin-bottom: 0.8rem;'>🔍</div>
                    <h4 style='margin: 0.5rem 0; color: #2c3e50;'>Search Symptoms</h4>
                    <p style='color: #6c757d; font-size: 0.9rem; margin: 0;'>Quickly find your symptoms</p>
                </div>
                <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 1.8rem; margin-bottom: 0.8rem;'>🤖</div>
                    <h4 style='margin: 0.5rem 0; color: #2c3e50;'>AI Analysis</h4>
                    <p style='color: #6c757d; font-size: 0.9rem; margin: 0;'>Get instant insights</p>
                </div>
                <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 1.8rem; margin-bottom: 0.8rem;'>💡</div>
                    <h4 style='margin: 0.5rem 0; color: #2c3e50;'>Personalized Advice</h4>
                    <p style='color: #6c757d; font-size: 0.9rem; margin: 0;'>Tailored recommendations</p>
                </div>
            </div>
            
            <div style='margin: 3rem 0 1rem; padding: 1.5rem; background: #f0f7f4; border-radius: 10px;'>
                <h4 style='margin-top: 0; color: #2c3e50;'>ℹ️ Important Note</h4>
                <p style='color: #5a6268; margin-bottom: 0; font-size: 0.9rem;'>
                    This tool is for informational purposes only and is not a substitute for professional 
                    medical advice, diagnosis, or treatment. Always seek the advice of your physician 
                    or other qualified health provider with any questions you may have.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Quick symptom buttons
        st.markdown("### Common Symptoms")
        common_symptoms = ["fever", "headache", "cough", "fatigue", "nausea", "dizziness"]
        cols = st.columns(3)
        for i, symptom in enumerate(common_symptoms):
            with cols[i % 3]:
                if st.button(symptom.title(), use_container_width=True, 
                           type="secondary" if symptom not in st.session_state.selected_symptoms else "primary"):
                    st.session_state.selected_symptoms[symptom] = 1
                    st.rerun()
    
    else:
        # Main content area with selected symptoms
        st.markdown("### Selected Symptoms")
        
        # Show selected symptoms as chips
        selected_chips = ""
        for symptom in selected_symptoms:
            selected_chips += (
                f"<span style='display: inline-block; background: #e3f2fd; color: #1976d2; "
                f"padding: 0.3rem 0.8rem; border-radius: 20px; margin: 0.2rem; font-size: 0.9rem;'>"
                f"{symptom.replace('_', ' ').title()}"
                f"<a href='#' onclick='event.stopPropagation(); return false;' "
                f"style='margin-left: 8px; color: #1976d2; text-decoration: none; font-weight: bold;' "
                f"onmouseover='this.style.textDecoration=\"underline\"' "
                f"onmouseout='this.style.textDecoration=\"none\"' "
                f"data-symptom='{symptom}'>×</a>"
                "</span>"
            )
        
        st.markdown(f"<div style='margin-bottom: 1.5rem;'>{selected_chips}</div>", unsafe_allow_html=True)
        
        # Add JavaScript to handle chip removal
        st.markdown("""
            <script>
            document.addEventListener('click', function(e) {
                if (e.target && e.target.dataset.symptom) {
                    const symptom = e.target.dataset.symptom;
                    const url = new URL(window.location.href);
                    url.searchParams.set('remove_symptom', symptom);
                    window.history.pushState({}, '', url);
                    window.location.reload();
                }
            });
            </script>
        """, unsafe_allow_html=True)
        
        # Check if a symptom should be removed
        if 'remove_symptom' in st.query_params:
            symptom_to_remove = st.query_params['remove_symptom']
            if symptom_to_remove in st.session_state.selected_symptoms:
                del st.session_state.selected_symptoms[symptom_to_remove]
                st.rerun()
        
        # Analyze button
        if st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True):
            with st.spinner("Analyzing your symptoms..."):
                # Generate report
                report = engine.generate_comprehensive_report(selected_symptoms)
                st.session_state.report = report
                st.rerun()
        
        # Display results if available
        if st.session_state.report:
            report = st.session_state.report
            
            # Display all sections with better spacing
            st.markdown("---")
            display_prediction_results(report)
            
            st.markdown("---")
            # Display active symptoms directly without expander
            display_active_symptoms(report)
            
            st.markdown("---")
            display_disease_description(report)
            
            st.markdown("---")
            display_recommendations(report)
            
            # Download report button
            st.markdown("---")
            with st.expander("💾 Download Report", expanded=False):
                # Create text report
                text_report = f"""
HEALTH ASSESSMENT REPORT
{'=' * 80}

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

PREDICTED CONDITION
- {report['prediction']['disease']} ({report['prediction']['confidence']:.1%} confidence)
- Severity: {report['severity']['level']} (Score: {report['severity']['score']}/7)

SYMPTOMS
{chr(10).join(f"- {s.replace('_', ' ').title()}" for s in report['symptoms']['active_symptoms'])}

OVERVIEW
{report['description']}

RECOMMENDATIONS

MEDICATIONS
{chr(10).join(f"- {m}" for m in report['medications']) if report['medications'] else "- No specific medication recommendations"}

DIET & NUTRITION
{chr(10).join(f"- {d}" for d in report['diet']) if report['diet'] else "- No specific diet recommendations"}

LIFESTYLE
{chr(10).join(f"- {w}" for w in report['workouts'][:5]) if report['workouts'] else "- No specific lifestyle recommendations"}

PRECAUTIONS
{chr(10).join(f"- {p}" for p in report['precautions']) if report['precautions'] else "- No specific precautions"}

{'=' * 80}
IMPORTANT: This is an AI-generated assessment for informational purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.
Always seek the advice of your physician or other qualified health provider with
any questions you may have regarding a medical condition.
{'=' * 80}
                """.format(
                    pd=pd,
                    report=report,
                    chr=chr
                )
                
                st.download_button(
                    label="💾 Download Report",
                    data=text_report,
                    file_name=f"health_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Disclaimer
            st.markdown("---")
            st.markdown("""
                <div style='background: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107;'>
                    <p style='margin: 0; color: #856404; font-size: 0.9rem;'>
                        <strong>Medical Disclaimer:</strong> This tool is for informational purposes only and 
                        is not a substitute for professional medical advice, diagnosis, or treatment. 
                        Always seek the advice of your physician or other qualified health provider 
                        with any questions you may have regarding a medical condition.
                    </p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
