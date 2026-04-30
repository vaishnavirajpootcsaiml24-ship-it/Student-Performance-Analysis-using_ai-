import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def generate_ai_feedback(grade, study_hours, attendance, participation, confidence=0.0):
    """
    Generates structured AI feedback based on the predicted grade and student metrics.
    Returns a dictionary with summary, strengths, improvements, actions, and confidence_note.
    """
    feedback = {
        "summary": "",
        "strengths": [],
        "improvements": [],
        "actions": [],
        "confidence_note": f"This prediction is based on a model confidence of {confidence*100:.1f}%. Values are derived from historical patterns."
    }

    if grade == "A":
        feedback["summary"] = "Excellent performance! You are demonstrating a strong grasp of the material and consistent effort."
        feedback["strengths"].append("Outstanding academic performance and consistency.")
        if attendance >= 90: feedback["strengths"].append("Perfect or near-perfect attendance record.")
        if study_hours >= 20: feedback["strengths"].append("Exceptional commitment to independent study.")
        
        feedback["actions"].append("Consider mentoring peers to reinforce your own knowledge.")
        feedback["actions"].append("Explore advanced topics or extracurricular research in your field.")

    elif grade == "B":
        feedback["summary"] = "Good job! You're performing well, but there's room to reach the top tier."
        feedback["strengths"].append("Solid understanding of the core concepts.")
        if participation > 7: feedback["strengths"].append("Active classroom engagement.")
        
        feedback["improvements"].append("Slight gaps in consistency might be preventing an 'A' grade.")
        feedback["actions"].append("Review the topics where you lost marks in recent assignments.")
        feedback["actions"].append("Try to increase study focus by 2-3 hours per week.")

    elif grade == "C":
        feedback["summary"] = "Average performance. You are meeting basic requirements but need more effort to excel."
        if attendance > 80: feedback["strengths"].append("Regular attendance is a good foundation.")
        
        feedback["improvements"].append("Study time may be insufficient for deep understanding.")
        feedback["improvements"].append("Class participation could be improved.")
        
        feedback["actions"].append("Create a more structured weekly study schedule.")
        feedback["actions"].append("Ask more questions during lectures to clarify doubts.")
        feedback["actions"].append("Seek help from teaching assistants for challenging topics.")

    elif grade == "D":
        feedback["summary"] = "Below average. Your current metrics suggest you are at risk of falling behind."
        feedback["improvements"].append("Significant lack of study hours or attendance.")
        feedback["improvements"].append("Low engagement with the course material.")
        
        feedback["actions"].append("Prioritize attendance to catch up on missed concepts.")
        feedback["actions"].append("Dedicate at least 10 more hours per week to focused study.")
        feedback["actions"].append("Join a study group for collaborative learning.")

    else: # Grade F
        feedback["summary"] = "Critical performance level. Immediate and drastic changes are required to pass."
        feedback["improvements"].append("Inadequate preparation and attendance.")
        feedback["improvements"].append("High risk of academic failure.")
        
        feedback["actions"].append("Schedule an urgent meeting with your academic advisor.")
        feedback["actions"].append("Eliminate distractions and focus entirely on core subjects.")
        feedback["actions"].append("Submit all pending assignments immediately.")

    return feedback

class DummyScaler:
    """A dummy scaler to replace StandardScaler since tree-based models don't need scaling.
       This prevents app.py from breaking when it expects a scaler."""
    def transform(self, data):
        return data

def main():
    print("Loading dataset...")
    # 1. Load dataset using pandas
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'students_performance.csv')
    df = pd.read_csv(data_path)
    
    # 2. Perform data cleaning (Warning-free method)


    print("Cleaning data...") 
    # ✅ Step 1: Handle target FIRST (important)
    grade_mapping = {'A': 95, 'B': 85, 'C': 75, 'D': 65, 'F': 55}
    if 'grade' in df.columns:
        df['performance_score'] = df['grade'].map(grade_mapping)
        df = df.drop(columns=['grade'])

    # ✅ Step 2: Convert ONLY numeric columns
    numeric_cols = [
        'study_hours', 'attendance', 'participation', 'extra_curricular',
        'previous_gpa', 'assignment_completion', 'sleep_hours',
        'stress_level', 'motivation_score'
    ]

    # Step 1: Convert to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Step 2: Handle missing values
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # 3. Feature engineering: create interaction and ratio features
    df['engagement_score'] = (df['attendance'] / 100.0) * df['assignment_completion']
    df['stress_sleep_ratio'] = df['stress_level'] / (df['sleep_hours'] + 1.0)
    df['gpa_study_interaction'] = df['previous_gpa'] * df['study_hours']
    df['participation_efficiency'] = df['participation'] / (df['study_hours'] + 1.0)
    df['study_hours_squared'] = df['study_hours'] ** 2
    df['extra_curricular_high'] = (df['extra_curricular'] > df['extra_curricular'].median()).astype(int)

    # 4. Target Variable
    # Map 'grade' into a numerical performance score for regression training
    grade_mapping = {'A': 95, 'B': 85, 'C': 75, 'D': 65, 'F': 55}
    if 'grade' in df.columns:
        df['performance_score'] = df['grade'].map(grade_mapping)
        df = df.drop(columns=['grade'])
        
    features_to_keep = [
        'study_hours',
        'attendance',
        'participation',
        'extra_curricular',
        'previous_gpa',
        'assignment_completion',
        'sleep_hours',
        'stress_level',
        'motivation_score',
        'parent_education',
        'engagement_score',
        'stress_sleep_ratio',
        'gpa_study_interaction',
        'participation_efficiency',
        'study_hours_squared',
        'extra_curricular_high'
    ]

    # Drop any other columns not in features_to_keep (except the target)
    for col in df.columns:
        if col not in features_to_keep and col != 'performance_score':
            df = df.drop(columns=[col])
            
    print("Final Features used for training: ", features_to_keep)
    
    # 5. Generate Heatmap and Boxplot
    FRONTEND_STATIC = os.path.join(BASE_DIR, '..', 'frontend', 'static')
    os.makedirs(FRONTEND_STATIC, exist_ok=True)
    
    # Heatmap (Correlation Matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FRONTEND_STATIC, 'heatmap.png'))
    plt.close()
    
    # Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[features_to_keep])
    plt.title("Feature Distribution Boxplot")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FRONTEND_STATIC, 'boxplot.png'))
    plt.close()
    
    print(f"Visualizations saved to {FRONTEND_STATIC}")

    # 6. Separate features and target
    X = df[features_to_keep]
    y = df['performance_score']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Note: Tree-based models (like Random Forest or Gradient Boosting) do NOT require feature scaling.
    # We removed StandardScaler entirely to simplify the pipeline.
    
    # 7. Train a robust model: Gradient Boosting Regressor with hyperparameter tuning
    print("\nTraining Gradient Boosting Regressor with hyperparameter search...")
    param_grid = {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [3, 5]
    }
    base_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    print("Best hyperparameters:", grid_search.best_params_)
    print("Best CV MAE:", round(-grid_search.best_score_, 4))
    
    # Train best estimator on the full training fold again to ensure final model is ready
    model.fit(X_train, y_train)
    
    # 8. Model Evaluation
    # Predict on training set
    train_preds = model.predict(X_train)
    train_r2 = r2_score(y_train, train_preds)
    
    # Predict on test set
    test_preds = model.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    
    # Print accuracy strictly in the terminal
    print("-" * 30)
    print("MODEL EVALUATION")
    print("-" * 30)
    print(f"Train R2 Score: {round(train_r2, 4)}")
    print(f"Test R2 Score:  {round(test_r2, 4)}")
    print(f"MAE (Error):    {round(test_mae, 4)}")
    print("-" * 30)
    
    # 9. Save trained model using pickle
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\nSaving Gradient Boosting Regressor to {MODELS_DIR} directory...")
    with open(os.path.join(MODELS_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
        
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(DummyScaler(), f)
        
    print("Model training completed successfully!")

if __name__ == '__main__':
    main() 
    