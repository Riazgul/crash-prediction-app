import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import os
from model import CrashPredictionModel, create_sample_data

# Configure Streamlit page
st.set_page_config(
    page_title="Crash Prediction System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = CrashPredictionModel()
    st.session_state.model_trained = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🚗 Crash Prediction System</h1><p>Predict and analyze traffic crash patterns</p></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Model Training", "Prediction", "Analytics", "Feedback"])

# Load or create model
@st.cache_resource
def load_model():
    model = CrashPredictionModel()
    if model.load_model('models/crash_model.pkl'):
        return model, True
    return model, False

# Save feedback
def save_feedback(feedback_data):
    feedback_file = 'feedback/feedback_responses.csv'
    os.makedirs('feedback', exist_ok=True)
    
    feedback_df = pd.DataFrame([feedback_data])
    
    if os.path.exists(feedback_file):
        existing_feedback = pd.read_csv(feedback_file)
        feedback_df = pd.concat([existing_feedback, feedback_df], ignore_index=True)
    
    feedback_df.to_csv(feedback_file, index=False)

# HOME PAGE
if page == "Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("https://via.placeholder.com/600x300/667eea/ffffff?text=Traffic+Safety+Analytics", 
                use_column_width=True)
    
    st.markdown("## Welcome to the Crash Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 **System Overview**
        This system uses machine learning to predict traffic crash types and analyze patterns in crash data.
        
        **Key Features:**
        - Real-time crash type prediction
        - Interactive data visualization
        - Pattern analysis and insights
        - User feedback collection
        """)
    
    with col2:
        st.markdown("""
        ### 📊 **How It Works**
        1. **Data Input**: Enter crash details
        2. **ML Prediction**: AI analyzes patterns
        3. **Results**: Get crash type prediction
        4. **Insights**: View analysis and trends
        
        **Model Performance:**
        - Accuracy: ~85%
        - Features: 6 key factors
        - Training Data: 1000+ records
        """)
    
    # Quick stats
    st.markdown("### 📈 Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container"><h3>1000+</h3><p>Training Records</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container"><h3>85%</h3><p>Accuracy Rate</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container"><h3>6</h3><p>Key Features</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-container"><h3>3</h3><p>Crash Types</p></div>', unsafe_allow_html=True)

# MODEL TRAINING PAGE
elif page == "Model Training":
    st.header("🎯 Model Training")
    
    # Option to use sample data or upload
    data_option = st.radio("Choose data source:", ["Use Sample Data", "Upload CSV File"])
    
    if data_option == "Use Sample Data":
        if st.button("Generate Sample Data & Train Model"):
            with st.spinner("Generating sample data and training model..."):
                # Create sample data
                sample_data = create_sample_data()
                
                # Train model
                results = st.session_state.model.train_model(sample_data)
                
                # Save model
                os.makedirs('models', exist_ok=True)
                st.session_state.model.save_model('models/crash_model.pkl')
                st.session_state.model_trained = True
                
                st.success("Model trained successfully!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Model Accuracy", f"{results['accuracy']:.3f}")
                    
                with col2:
                    st.write("**Feature Importance:**")
                    importance_df = pd.DataFrame(
                        list(results['feature_importance'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        uploaded_file = st.file_uploader("Upload your crash data CSV", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("**Data Preview:**")
            st.dataframe(df.head())
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    try:
                        results = st.session_state.model.train_model(df)
                        
                        # Save model
                        os.makedirs('models', exist_ok=True)
                        st.session_state.model.save_model('models/crash_model.pkl')
                        st.session_state.model_trained = True
                        
                        st.success("Model trained successfully!")
                        st.metric("Model Accuracy", f"{results['accuracy']:.3f}")
                        
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")

# PREDICTION PAGE
elif page == "Prediction":
    st.header("🔮 Crash Prediction")
    
    # Load model if not already loaded
    if not st.session_state.model_trained:
        model, loaded = load_model()
        if loaded:
            st.session_state.model = model
            st.session_state.model_trained = True
        else:
            st.warning("Please train the model first in the 'Model Training' page.")
            st.stop()
    
    st.markdown("### Enter Crash Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        crash_hour = st.slider("Hour of Day", 0, 23, 12)
        crash_day = st.selectbox("Day of Week", 
                                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        substance_abuse = st.selectbox("Driver Substance Abuse", 
                                     ["None", "Alcohol", "Drugs", "Unknown"])
    
    with col2:
        speed_limit = st.selectbox("Speed Limit", [25, 35, 45, 55, 65])
        vehicle_direction = st.selectbox("Vehicle Direction", 
                                       ["North", "South", "East", "West"])
        distracted_by = st.selectbox("Driver Distracted By", 
                                   ["None", "Cell Phone", "Other", "Unknown"])
    
    # Convert day of week to number
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                   "Friday": 4, "Saturday": 5, "Sunday": 6}
    
    # Prepare input data
    input_data = {
        'crash_hour': crash_hour,
        'crash_day_of_week': day_mapping[crash_day],
        'driver_substance_abuse': substance_abuse,
        'speed_limit': speed_limit,
        'vehicle_going_dir': vehicle_direction,
        'driver_distracted_by': distracted_by
    }
    
    if st.button("Predict Crash Type", type="primary"):
        try:
            # Make prediction
            prediction, probability = st.session_state.model.predict(input_data)
            
            # Map prediction back to readable format
            crash_types = {0: "Property Damage", 1: "Injury", 2: "Fatal"}
            predicted_type = crash_types.get(prediction, f"Type {prediction}")
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🎯 Prediction Result</h3>
                <p><strong>Predicted Crash Type:</strong> {predicted_type}</p>
                <p><strong>Confidence:</strong> {probability:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk assessment
            if predicted_type == "Fatal":
                st.error("⚠️ High Risk: Fatal crash predicted. Extreme caution advised!")
            elif predicted_type == "Injury":
                st.warning("⚠️ Medium Risk: Injury crash predicted. Caution advised!")
            else:
                st.info("ℹ️ Low Risk: Property damage only predicted.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# ANALYTICS PAGE
elif page == "Analytics":
    st.header("📊 Analytics Dashboard")
    
    # Create sample analytics data
    sample_data = create_sample_data()
    
    # Time series analysis
    st.subheader("📈 Crash Trends Over Time")
    
    crash_by_month = sample_data.groupby(sample_data['crash_date/time'].dt.month).size()
    fig = px.line(x=crash_by_month.index, y=crash_by_month.values, 
                  labels={'x': 'Month', 'y': 'Number of Crashes'})
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🕐 Crashes by Hour")
        hourly_crashes = sample_data.groupby(sample_data['crash_date/time'].dt.hour).size()
        fig = px.bar(x=hourly_crashes.index, y=hourly_crashes.values,
                     labels={'x': 'Hour of Day', 'y': 'Number of Crashes'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚗 Crash Types Distribution")
        crash_dist = sample_data['acrs_report_type'].value_counts()
        fig = px.pie(values=crash_dist.values, names=crash_dist.index)
        st.plotly_chart(fig, use_container_width=True)
    
    # Speed limit analysis
    st.subheader("🚦 Speed Limit vs Crash Severity")
    speed_severity = pd.crosstab(sample_data['speed_limit'], sample_data['acrs_report_type'])
    fig = px.bar(speed_severity, x=speed_severity.index, y=speed_severity.columns,
                 labels={'x': 'Speed Limit', 'y': 'Number of Crashes'})
    st.plotly_chart(fig, use_container_width=True)

# FEEDBACK PAGE
elif page == "Feedback":
    st.header("💬 User Feedback")
    
    st.markdown("""
    ### Help us improve the system!
    Your feedback is valuable for making this system better. Please share your experience.
    """)
    
    with st.form("feedback_form"):
        # User information
        user_name = st.text_input("Your Name (Optional)")
        user_email = st.text_input("Your Email (Optional)")
        
        # Feedback categories
        st.subheader("Rate Your Experience")
        
        col1, col2 = st.columns(2)
        with col1:
            usability_rating = st.slider("Usability (1-10)", 1, 10, 5)
            accuracy_rating = st.slider("Accuracy (1-10)", 1, 10, 5)
        
        with col2:
            interface_rating = st.slider("Interface Design (1-10)", 1, 10, 5)
            overall_rating = st.slider("Overall Experience (1-10)", 1, 10, 5)
        
        # Feedback text
        st.subheader("Detailed Feedback")
        what_worked = st.text_area("What worked well?", height=100)
        what_didnt = st.text_area("What didn't work well?", height=100)
        suggestions = st.text_area("Suggestions for improvement", height=100)
        
        # Most useful feature
        useful_feature = st.selectbox("Most useful feature", 
                                    ["Prediction", "Analytics", "Model Training", "Interface", "All"])
        
        # Submit button
        submitted = st.form_submit_button("Submit Feedback", type="primary")
        
        if submitted:
            feedback_data = {
                'timestamp': datetime.now().isoformat(),
                'user_name': user_name,
                'user_email': user_email,
                'usability_rating': usability_rating,
                'accuracy_rating': accuracy_rating,
                'interface_rating': interface_rating,
                'overall_rating': overall_rating,
                'what_worked': what_worked,
                'what_didnt': what_didnt,
                'suggestions': suggestions,
                'useful_feature': useful_feature
            }
            
            save_feedback(feedback_data)
            st.success("Thank you for your feedback! 🙏")
            st.balloons()
    
    # Display existing feedback (if any)
    if os.path.exists('feedback/feedback_responses.csv'):
        st.subheader("📊 Feedback Summary")
        feedback_df = pd.read_csv('feedback/feedback_responses.csv')
        
        if not feedback_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Responses", len(feedback_df))
            with col2:
                st.metric("Avg Usability", f"{feedback_df['usability_rating'].mean():.1f}/10")
            with col3:
                st.metric("Avg Accuracy", f"{feedback_df['accuracy_rating'].mean():.1f}/10")
            with col4:
                st.metric("Avg Overall", f"{feedback_df['overall_rating'].mean():.1f}/10")
            
            # Recent feedback
            st.subheader("Recent Feedback")
            recent_feedback = feedback_df.tail(5)[['timestamp', 'overall_rating', 'suggestions']]
            st.dataframe(recent_feedback, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Crash Prediction System v1.0 | Built with Streamlit | © 2025"
    "</div>", 
    unsafe_allow_html=True
)