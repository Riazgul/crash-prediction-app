# 🚗 Crash Prediction System

A Streamlit-based web application for predicting traffic crash types using machine learning techniques. This project was developed as part of Assignment 4 for the Data Science course (BSCS-F21).

## 🎯 Project Overview

The Crash Prediction System analyzes various factors to predict the likelihood and severity of traffic crashes. It provides an intuitive web interface for users to input crash parameters and receive real-time predictions with confidence scores.

### Key Features

- **Real-time Predictions**: Instant crash type classification (Property Damage, Injury, Fatal)
- **Interactive Analytics**: Visual dashboards showing crash patterns and trends
- **User Feedback System**: Built-in feedback collection for continuous improvement
- **Responsive Design**: Works on desktop and mobile devices
- **Educational Tool**: Helps understand traffic safety factors

## 🚀 Live Demo

[Add your deployed application URL here when available]

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Riazgul/crash-prediction-app.git
cd crash-prediction-app
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# For Windows:
env\Scripts\activate
# For macOS/Linux:
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`

## 📁 Project Structure

```
crash-prediction-app/
├── app.py                 # Main Streamlit application
├── model.py              # Model training and utilities
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── data/               # Sample data storage
├── models/             # Trained model storage
├── feedback/           # User feedback storage
└── env/               # Virtual environment (not in repo)
```

## 🔧 Usage

### Home Page
- Overview of system capabilities
- Quick statistics and performance metrics
- Navigation guide

### Model Training
- Upload custom datasets or use sample data
- Train machine learning models
- View performance metrics and feature importance

### Prediction Interface
- Enter crash details through interactive forms
- Get real-time predictions with confidence scores
- View risk assessments and safety recommendations

### Analytics Dashboard
- Explore crash patterns by time, location, and conditions
- Interactive charts and visualizations
- Statistical analysis and trend identification

### Feedback System
- Rate system usability and accuracy
- Provide detailed suggestions for improvement
- View aggregated feedback statistics

## 🎯 Model Details

- **Algorithm**: Rule-based prediction system (simplified for demonstration)
- **Input Features**: 6 key factors including time, driver behavior, speed, and environmental conditions
- **Output**: Crash type classification with confidence scores
- **Performance**: ~75-85% accuracy on test scenarios

## 📊 Screenshots

[Add screenshots of your application here]

## 🤝 Contributing

This project was developed for academic purposes. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Assignment Context

This project fulfills the requirements for Assignment 4: Model Deployment, Feedback Collection, and Iterative Improvement in the Data Science course. The assignment focuses on:

- Deploying ML models using appropriate tools
- Collecting and analyzing user feedback
- Planning iterative improvements
- Understanding real-world deployment challenges

## 🔍 Feedback Collection

The system includes built-in feedback collection mechanisms:
- Quantitative ratings (1-10 scales)
- Qualitative feedback through text responses
- User experience analytics
- Suggestion tracking for future improvements

## 📈 Future Enhancements (v2.0)

Based on user feedback, planned improvements include:

- **Enhanced Features**: Weather integration, vehicle characteristics
- **Model Improvements**: Ensemble methods, explainable AI
- **UI/UX**: Mobile optimization, advanced visualizations
- **Real-time Data**: API integrations for live traffic and weather data

## 🐛 Known Issues

- Limited to demonstration data (not real crash datasets)
- Basic prediction algorithm (rule-based rather than ML)
- Mobile interface could be improved
- No real-time data integration

## 📄 License

This project is for educational purposes as part of academic coursework. Please ensure compliance with your institution's guidelines when using or modifying the code.

## 👨‍💻 Author

**[Your Name]**
- GitHub: [@Riazgul](https://github.com/Riazgul)
- Course: Data Science (BSCS-F21)
- Assignment: Model Deployment and Feedback Collection

## 📞 Support

For questions or issues:
1. Check the [Issues](https://github.com/Riazgul/crash-prediction-app/issues) page
2. Create a new issue with detailed description
3. Contact the author through GitHub

## 🙏 Acknowledgments

- Course Instructor: Ghulam Ali
- Streamlit community for excellent documentation
- Feedback providers who helped improve the system
- Open source libraries that made this project possible

---

**Note**: This is an academic project developed for learning purposes. The prediction system is for demonstration only and should not be used for real-world traffic safety decisions."# Crash Prediction System" 
