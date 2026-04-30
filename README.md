# 🎓 Student Performance Prediction using AI
### *Empowering Education with Data-Driven Insights*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)

---

## 📌 Overview
Predicting how a student will perform is more than just looking at their past grades. This project uses Artificial Intelligence to analyze various factors like study habits, attendance, and even sleep patterns to forecast academic outcomes. 

The goal is to provide a tool that helps students and teachers identify areas for improvement early on, ensuring better success in the long run.

## 🚀 Features
- **Smart Predictions**: Get an AI-predicted grade (A-F) based on your daily routine and academic habits.
- **Personalized Feedback**: The system doesn't just give a grade; it provides actionable advice on how to improve.
- **Visual Analytics**: Interactive charts to help you understand your performance trends over time.
- **Secure Access**: A clean login and registration system with built-in security features like CSRF protection and rate limiting.
- **History Tracking**: Keep a record of all your past predictions to see how your habits affect your results.

## 🧠 How the system works
1. **Input Data**: The user enters details like study hours, attendance percentage, previous GPA, and lifestyle factors (sleep, stress, etc.).
2. **Data Processing**: The backend takes this raw data and calculates "engineered features" like *Study Efficiency* and *Engagement Score*.
3. **ML Model Inference**: The processed data is fed into a trained **Gradient Boosting** model.
4. **Result Generation**: The model predicts the most likely grade and calculates its confidence level.
5. **Output**: The dashboard displays the predicted grade, a breakdown of the confidence scores, and custom AI-generated feedback.

## 🛠️ Tech Stack
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript (Chart.js for analytics)
- **Testing**: Pytest

## 📁 Project Structure
```text
Student_performance_using_AI/
├── backend/
│   ├── app.py
│   ├── src/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── predict.py
│   │   └── train_model.py
│   ├── models/
│   │   ├── model.pkl
│   │   └── scaler.pkl
│   ├── data/
│   │   └── student_performance.csv
│   ├── database.db
│   ├── output.txt
│   ├── test_app.py
│   └── test_db.py
├── frontend/
│   ├── templates/
│   │   ├── dashboard.html
│   │   ├── dashboard_template.txt
│   │   ├── index.html
│   │   ├── login.html
│   │   ├── profile.html
│   │   └── register.html
│   └── static/
│       ├── boxplot.png
│       ├── feature_importance.png
│       └── heatmap.png
├── .gitignore
├── README.md
└── requirements.txt
```

## ⚙️ Installation Steps
1. **Clone the project**:
   ```bash
   git clone https://github.com/yourusername/Student_performance_using_AI.git
   cd Student_performance_using_AI
   ```
2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ How to Run the Project
1. Navigate to the root directory.
2. Run the Flask application:
   ```bash
   python backend/app.py
   ```
3. Open your browser and go to: `http://127.0.0.1:5000`

## 🌍 Deployment
This project is ready to be deployed to platforms like **Render**, **Heroku**, or any standard Linux server. 

**Deployment Prerequisites included:**
- `Procfile`: Configured for Gunicorn.
- `requirements.txt`: Updated with `gunicorn` and `python-dotenv`.
- `runtime.txt`: Specifies Python 3.11.9.
- `.env.example`: Template for required environment variables.

**Example: Deploying to Render**
1. Create a new Web Service on Render and link your GitHub repository.
2. Set the Build Command to: `pip install -r requirements.txt`
3. Set the Start Command to: `gunicorn backend.app:app`
4. Add environment variables:
   - `SECRET_KEY`: Generate a secure random string.
   - `FLASK_ENV`: Set to `production`.
   - `DB_PATH`: Optionally mount a disk to `/var/data` and set DB_PATH to `/var/data/database.db` to persist user history.

## 🧪 Testing
The project includes automated tests to ensure everything is working correctly.
- **App Tests**: `pytest backend/test_app.py` (Checks routes, login, and predictions)
- **Database Tests**: `python backend/test_db.py` (Ensures the database is correctly saving users and history)

## 🤖 ML Model Explanation
The project uses a **Gradient Boosting Regressor**, which is an advanced machine learning algorithm that builds multiple small decision trees to minimize errors. 
- **Training**: The model was trained on historical student data to learn the relationship between habits and grades.
- **Optimization**: We used *GridSearchCV* to find the best settings (hyperparameters) for the model, ensuring high accuracy.
- **Why this model?**: It handles complex data relationships very well and provides reliable predictions even with small variations in input.

## 📊 Dataset Information
The system is powered by a dataset containing various student metrics:
- **Academic**: Attendance, Study Hours, Previous GPA, Assignment Completion.
- **Lifestyle**: Sleep Hours, Stress Level, Extra-curricular Participation.
- **Demographic**: Parent Education Level.
This data allows the AI to form a holistic view of a student's academic environment.

## 📸 Screenshots Section
*Here is a look at the application in action:*

| **Dashboard View** | **Prediction Results** |
|:---:|:---:|
| ![Dashboard Placeholder](https://via.placeholder.com/400x250?text=User+Dashboard) | ![Result Placeholder](https://via.placeholder.com/400x250?text=AI+Prediction+Result) |

| **Analytics & Trends** | **Login Page** |
|:---:|:---:|
| ![Analytics Placeholder](https://via.placeholder.com/400x250?text=Performance+Analytics) | ![Login Placeholder](https://via.placeholder.com/400x250?text=Secure+Login) |

## 🚀 Future Scope
- **Real-time Notifications**: Alerting students when their attendance or performance dips below a certain threshold.
- **More Data Points**: Integrating with school management systems for real-time grade tracking.
- **Multi-Class Predictions**: Expanding beyond simple grades to predict specific subject-wise performance.
- **Mobile App**: Developing a companion mobile app for easy access on the go.

---
*Developed with ❤️ for a better future in education.*
