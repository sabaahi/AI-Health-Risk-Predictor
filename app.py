import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import plotly.express as px
from io import StringIO

st.set_page_config(
    page_title="AI-Powered Health Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
    <style>
    /* Main sidebar background */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1.5rem;
    }
    
    /* All sidebar widgets container */
    .sidebar .stSelectbox,
    .sidebar .stSlider,
    .sidebar .stRadio,
    .sidebar .stNumberInput {
        background-color: white;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 15px;
        border: 1px solid #dee2e6;
    }
    
    /* Widget labels */
    .sidebar .stSelectbox label,
    .sidebar .stSlider label,
    .sidebar .stRadio label,
    .sidebar .stNumberInput label {
        color: #212529 !important;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Selectbox dropdown */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        color: #212529 !important;
    }
    
    /* Selectbox dropdown options */
    div[role="listbox"] div {
        background-color: white !important;
        color: #212529 !important;
    }
    
    /* Slider track */
    div[data-baseweb="slider"] > div > div > div > div {
        background-color: #6c757d !important;
    }
    
    /* Slider thumb */
    div[data-baseweb="slider"] > div > div > div > div > div {
        background-color: #495057 !important;
        border-color: #495057 !important;
    }
    
    /* Radio buttons container */
    div[data-baseweb="radio"] > div {
        background-color: white !important;
        color: #212529 !important;
    }
    
    /* Radio button selected */
    div[data-baseweb="radio"] > div > div:first-child > div {
        background-color: #495057 !important;
    }
    
    /* Header text color */
    .sidebar .stMarkdown h1, 
    .sidebar .stMarkdown h2, 
    .sidebar .stMarkdown h3 {
        color: #212529 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #495057;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
    }
    
    /* Button hover effect */
    .stButton>button:hover {
        background-color: #343a40;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# MAIN APP CONTENT
st.markdown(
    """
    <div style='text-align: center; margin-top: -30px;'>
        <h1 style='font-size: 40px; margin-bottom: 10px;'>🏥 AI-Powered Health Risk Predictor</h1>
        <p style='font-size: 18px;'>Upload your health dataset for analysis and predictive modeling</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load dataset
uploaded_file = st.file_uploader("Upload a Health Dataset (CSV)", type=["csv"], key="file_uploader")

def detect_target_column(df):
    """Automatically detect likely target column based on common names"""
    candidates = ['target', 'Outcome', 'outcome', 'class', 'stroke', 'diabetes', 
                 'disease', 'status', 'result', 'diagnosis', 'label']
    for col in df.columns:
        if col.lower() in [c.lower() for c in candidates]:
            return col
    # If no exact match, look for binary columns
    for col in df.columns:
        if len(df[col].unique()) == 2:
            return col
    return None

def smart_eda_visualizations(df, target_col):
    """Select the most appropriate visualizations based on data characteristics"""
    visualizations = []
    
    # 1. Target distribution (always show)
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    if len(df[target_col].unique()) <= 10:
        sns.countplot(x=target_col, data=df, ax=ax1, palette='viridis')
        ax1.set_title("Distribution of Target Classes")
        visualizations.append(("Target Distribution", fig1))
    
    # 2. Correlation heatmap (for numeric data)
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                   center=0, ax=ax2, mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
        ax2.set_title("Feature Correlation Heatmap")
        visualizations.append(("Correlation Heatmap", fig2))
    
    # 3. Pairplot for small datasets (<= 8 numeric features)
    if len(numeric_cols) <= 8 and len(numeric_cols) > 1:
        fig3 = sns.pairplot(df[numeric_cols].sample(min(200, len(df))), 
                           diag_kind='kde', corner=True)
        fig3.fig.suptitle("Feature Relationships", y=1.02)
        visualizations.append(("Feature Relationships", fig3))
    
    # 4. Boxplot for top 3 most important features vs target
    if target_col in numeric_cols:
        numeric_cols = numeric_cols.drop(target_col)
    if len(numeric_cols) > 0:
        top_features = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        for feat in top_features:
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            if len(df[target_col].unique()) <= 5:
                sns.boxplot(x=target_col, y=feat, data=df, ax=ax4, palette='pastel')
            else:
                sns.scatterplot(x=target_col, y=feat, data=df.sample(min(200, len(df))), ax=ax4)
            ax4.set_title(f"{feat} vs Target")
            visualizations.append((f"{feat} Distribution", fig4))
    
    return visualizations

def preprocess_data(df, target_col, test_size=0.2):
    """Handle data preprocessing including encoding and scaling"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode non-numeric target
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Handle non-numeric features
    non_numeric = X.select_dtypes(exclude=np.number).columns
    if len(non_numeric) > 0:
        X = pd.get_dummies(X, columns=non_numeric, drop_first=True)
    else:
        X = X.select_dtypes(include=np.number)
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, X.columns

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Basic data cleaning
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Display basic info
        st.subheader("Dataset Overview")
        buffer = StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            st.write(f"**Missing Values:** {df.isna().sum().sum()} (already removed)")
        with col2:
            st.write(f"**Numeric Columns:** {len(df.select_dtypes(include=np.number).columns)}")
            st.write(f"**Categorical Columns:** {len(df.select_dtypes(exclude=np.number).columns)}")
        
        # Target selection
        st.sidebar.header("Model Configuration")
        auto_target = detect_target_column(df)
        target_col = st.sidebar.selectbox(
            "Select Target Column", 
            df.columns, 
            index=df.columns.get_loc(auto_target) if auto_target else 0,
            key="target_column_select"
        )
        
        # Show smart EDA visualizations
        st.subheader("Smart Data Analysis")
        with st.expander("Show Data Analysis Visualizations", expanded=False):
            visualizations = smart_eda_visualizations(df, target_col)
            for name, fig in visualizations:
                st.pyplot(fig)
                plt.close()
        
        # Preprocess data
        test_size = st.sidebar.slider(
            "Test Size (%)", 
            10, 40, 20,
            key="test_size_slider"
        ) / 100
        
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df, target_col, test_size)
        
        # Model selection
        model_options = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Naive Bayes": GaussianNB()
        }
        model_choice = st.sidebar.radio(
            "Choose Model", 
            list(model_options.keys()),
            key="model_choice_radio"
        )
        model = model_options[model_choice]
        
        # Feature selection
        if st.sidebar.checkbox("Enable Feature Selection", True, key="feature_selection_checkbox"):
            k = st.sidebar.slider(
                "Number of Features to Select", 
                min_value=2, 
                max_value=min(20, len(feature_names)), 
                value=min(5, len(feature_names)),
                key="feature_k_slider"
            )
            selector = SelectKBest(f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            selected_features = feature_names[selector.get_support()]
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            selected_features = feature_names
        
        # Train model
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        y_prob = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Model evaluation
        st.subheader("Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
        col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2%}")
        
        # Confusion matrix
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, 
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=['Negative', 'Positive'], y=['Negative', 'Positive'])
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # ROC curve (if model supports probabilities)
        if y_prob is not None:
            st.write("**ROC Curve**")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = px.area(x=fpr, y=tpr, 
                             title=f'ROC Curve (AUC = {roc_auc:.2f})',
                             labels=dict(x='False Positive Rate', y='True Positive Rate'))
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc, use_container_width=True)
        
        # Feature importance
        st.write("**Feature Importance**")
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=selected_features)
            fig_fi = px.bar(importance.sort_values(ascending=True), 
                           orientation='h', 
                           title="Feature Importance Scores")
            st.plotly_chart(fig_fi, use_container_width=True)
        elif hasattr(model, "coef_"):
            coef = pd.Series(model.coef_[0], index=selected_features)
            fig_coef = px.bar(coef.sort_values(ascending=True), 
                             orientation='h', 
                             title="Feature Coefficients")
            st.plotly_chart(fig_coef, use_container_width=True)
        
        # Prediction interface
        st.sidebar.header("Live Prediction")
        st.sidebar.write(f"Using {len(selected_features)} features:")
        user_input = {}
        
        for i, feature in enumerate(selected_features):
            # Get reasonable min/max values for sliders
            col_min = df[feature].min() if feature in df.columns else X_train[feature].min()
            col_max = df[feature].max() if feature in df.columns else X_train[feature].max()
            default_val = df[feature].median() if feature in df.columns else X_train[feature].median()
            
            user_input[feature] = st.sidebar.slider(
                f"{feature}",
                min_value=float(col_min),
                max_value=float(col_max),
                value=float(default_val),
                step=float((col_max - col_min)/100),
                key=f"prediction_slider_{i}"
            )
        
        if st.sidebar.button("Predict", key="predict_button"):
            input_df = pd.DataFrame([user_input])
            # Scale the input data
            scaler = StandardScaler()
            scaler.fit(X_train[selected_features])
            input_scaled = scaler.transform(input_df)
            
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None
            
            if proba is not None:
                st.sidebar.success(
                    f"Prediction: {'Positive' if prediction == 1 else 'Negative'}\n"
                    f"Probability: {proba:.1%}"
                )
            else:
                st.sidebar.success(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
        
        # Data summary
        with st.expander("Show Dataset Summary", expanded=False):
            st.write(df.describe())
        
        # Download predictions
        if st.checkbox("Generate Predictions on Full Dataset", key="download_checkbox"):
            full_predictions = model.predict(X_test_selected)
            results_df = X_test.copy()
            results_df['Actual'] = y_test
            results_df['Predicted'] = full_predictions
            if y_prob is not None:
                results_df['Probability'] = model.predict_proba(X_test_selected)[:, 1]
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions",
                csv,
                "model_predictions.csv",
                "text/csv",
                key='download-csv'
            )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
        <p>Built with Streamlit | Developed by Sufyan Afzal and Muhammad Akbar</p>
    </div>
    """, unsafe_allow_html=True)