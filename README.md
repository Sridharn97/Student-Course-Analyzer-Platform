# Student Course Analyzer Platform

## Overview

The **Student Course Analyzer Platform** is a comprehensive data science–based dashboard built with Streamlit to evaluate and predict student performance in online courses. This platform provides educators and administrators with powerful analytics tools to understand student engagement, performance trends, and predictive insights using machine learning models.

## Features

### 📊 Main Dashboard
- **Dataset Preview**: Paginated view of student enrollment data
- **Key Metrics**: Average scores, course ratings, and progress percentages
- **Platform Analytics**: Compare performance across different learning platforms
- **Sentiment Analysis**: Analyze student feedback and sentiment trends
- **Performance Insights**: Visual representations of student achievements

### 🤖 ML Models Comparison
- Compare multiple machine learning models for score prediction:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost (if available)
  - Neural Network (MLP Regressor, if available)
- Cross-validation scores and performance metrics
- Feature importance analysis

### ⚠️ At-Risk Detection
- Identify students at risk of poor performance
- Risk assessment based on progress, engagement, and historical data
- Early intervention recommendations

### 🎯 Course Recommendations
- Personalized course recommendations based on student profiles
- Predictive analytics for course success likelihood

### 📄 Report Generator
- Generate comprehensive reports on student performance
- Exportable analytics and insights

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone or download the project files:
   ```
   git clone <repository-url>
   cd "Student Course Analyzer Platform"
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Dependencies
- streamlit
- pandas
- numpy
- scipy
- scikit-learn
- plotly
- seaborn
- matplotlib
- xgboost
- openpyxl

## Usage

1. Run the Streamlit application:
   ```
   streamlit run student_course_dashboard.py
   ```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`).

3. Upload your Excel data file:
   - Use the provided sample file `Course Analysis Prediction Dataset.xlsx` or your own data file
   - Required sheets: Courses, Students, Enrollments, Feedback, Platform_Performance

4. Navigate through different sections using the sidebar:
   - Select pages for different analyses
   - Upload and process data
   - View visualizations and predictions

## Workflow

### Step-by-Step Guide to Using the Platform

1. **Launch the Application**:
   - Run `streamlit run student_course_dashboard.py`
   - Open the provided URL in your browser

2. **Upload Data**:
   - Click on the file uploader in the sidebar
   - Select `Course Analysis Prediction Dataset.xlsx` (included in the project) or your own Excel file
   - The app will automatically process and merge the data from multiple sheets

3. **Explore the Main Dashboard**:
   - View dataset preview with pagination
   - Check key performance metrics (average scores, ratings, progress)
   - Analyze platform analytics, sentiment trends, and performance insights through interactive charts

4. **Compare ML Models**:
   - Navigate to "🤖 ML Models Comparison"
   - View performance metrics for different regression models
   - Analyze feature importance and cross-validation scores

5. **Identify At-Risk Students**:
   - Go to "⚠️ At-Risk Detection"
   - Review risk assessments based on student data
   - Get recommendations for early interventions

6. **Get Course Recommendations**:
   - Select "🎯 Course Recommendations"
   - Receive personalized suggestions based on student profiles

7. **Generate Reports**:
   - Use "📄 Report Generator" to create comprehensive performance reports
   - Export insights for further analysis

### Data Processing Flow
- **Input**: Excel file with student, course, enrollment, feedback, and platform data
- **Processing**: Data merging, missing value imputation, feature engineering
- **Analysis**: Statistical analysis, ML predictions, risk assessment
- **Output**: Interactive dashboards, visualizations, and downloadable reports

The application expects an Excel file with the following sheets:

- **Courses**: Course information including Course_ID, course details
- **Students**: Student profiles with Student_ID, demographics
- **Enrollments**: Enrollment records linking students to courses with scores, progress
- **Feedback**: Student feedback and sentiment data
- **Platform_Performance**: Platform-specific performance metrics

## Key Features Explained

### Data Processing
- Automatic data merging and preprocessing
- Handling missing values with appropriate imputation
- Categorical encoding for machine learning

### Machine Learning
- Predictive modeling for student scores
- Model comparison with R² scores and MSE
- Feature engineering and scaling

### Visualization
- Interactive charts using Plotly
- Seaborn and Matplotlib for statistical plots
- Real-time dashboard updates

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open-source. Please check the license file for details.

## Support

For issues or questions, please create an issue in the repository or contact the maintainers.</content>
<parameter name="filePath">d:\Projects\Student Course Analyzer Platform\README.md