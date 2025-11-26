# 🚀 Quick Start Guide

Get your Disease & Medicine Prediction System up and running in minutes!

## 🛠 Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## ⚡ Fast Track Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Medicine_Recommandation_System.git
cd Medicine_Recommandation_System
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python model_training.py
```
> 💡 When asked "Do you want to optimize the best model?", type `n` for quick start.

### 5. Launch the Application
```bash
streamlit run app.py
```

### 6. Access the Application
Open your web browser and navigate to:  
`http://localhost:8501`

## 🖥️ Using the System

1. **Select Symptoms**
   - Use the sidebar to search and select multiple symptoms
   - Adjust symptom severity if needed
   
2. **Get Diagnosis**
   - Click "Get Diagnosis & Recommendations"
   - View the predicted disease with confidence score
   
3. **Explore Recommendations**
   - Review medication suggestions
   - Check dietary recommendations
   - See lifestyle and precautionary measures
   
4. **View Detailed Report**
   - Scroll down for comprehensive health information
   - Check the severity assessment of your symptoms

## 🛠 Development Commands

### Run Data Exploration
```bash
python data_exploration.py
```

### Test Recommendation Engine
```bash
python recommendation_engine.py
```

### Run Tests
```bash
# Run unit tests
python -m pytest tests/

# Run with coverage report
coverage run -m pytest tests/
coverage report -m
```

## 🔧 Troubleshooting

### Module Not Found Error
```bash
# Make sure you're in the correct directory
cd path/to/Medicine_Recommandation_System

# Activate virtual environment
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Model Not Found Error
- Ensure you've run the model training script:
  ```bash
  python model_training.py
  ```
- Check that the `models/` directory contains the required `.pkl` files

### Streamlit Connection Issues
- If the app doesn't open automatically, try accessing it directly at `http://localhost:8501`
- If the port is in use, you can specify a different port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

## 📦 Deployment

### Local Deployment
1. Install gunicorn (if not using Streamlit Sharing)
   ```bash
   pip install gunicorn
   ```

2. Create a `Procfile` with:
   ```
   web: streamlit run app.py
   ```

### Cloud Deployment Options
- **Streamlit Sharing** (Recommended)
  1. Push your code to a GitHub repository
  2. Sign up at [Streamlit Sharing](https://share.streamlit.io/)
  3. Connect your repository and deploy

- **Heroku**
  ```bash
  # Install Heroku CLI
  heroku create your-app-name
  git push heroku main
  ```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
```bash
python model_training.py
```

### Issue: Streamlit won't start
**Solution**: Check if port 8501 is available
```bash
streamlit run app.py --server.port 8502
```

### Issue: Low accuracy
**Solution**: Optimize the model during training (choose 'y' when prompted)

## 💡 Tips for Best Results

1. **Select Multiple Symptoms**: More symptoms = better accuracy
2. **Be Specific**: Choose exact symptoms you're experiencing
3. **Use Search**: Type to quickly find symptoms
4. **Check Top-3**: Review alternative diagnoses
5. **Download Report**: Save for future reference

## 🎯 Example Usage

### Example 1: Fungal Infection
**Symptoms**: itching, skin_rash, nodal_skin_eruptions
**Expected**: Fungal infection (>95% confidence)

### Example 2: Diabetes
**Symptoms**: increased_appetite, polyuria, excessive_hunger, weight_loss
**Expected**: Diabetes (>90% confidence)

### Example 3: Common Cold
**Symptoms**: continuous_sneezing, runny_nose, cough, mild_fever
**Expected**: Common Cold (>85% confidence)

## 📊 What to Expect

### Model Training Output
- Model comparison table
- Best model selection
- Accuracy metrics
- Saved model files in `models/`

### Web Application Features
- 132 searchable symptoms
- Real-time predictions
- 4 types of recommendations
- Downloadable reports
- Interactive visualizations

## 🎓 Learning Path

1. **Beginner**: Use the web app to make predictions
2. **Intermediate**: Explore data with `data_exploration.py`
3. **Advanced**: Understand model with `model_explainability.py`
4. **Expert**: Modify and improve the models

## 📚 Next Steps

After quick start, explore:
- [ ] Read full README.md
- [ ] Review model performance report
- [ ] Check visualizations folder
- [ ] Try different symptom combinations
- [ ] Customize recommendations
- [ ] Contribute improvements

## ⚠️ Remember

This is an educational tool. Always consult healthcare professionals for medical advice!

---

**Need Help?** Check README.md for detailed documentation.
