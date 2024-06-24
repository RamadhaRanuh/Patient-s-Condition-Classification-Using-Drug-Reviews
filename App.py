import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import PIL
import os

# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

# Get current folder path
current_dir = os.path.dirname(__file__)
# print(current_folder)

@st.cache_data
def LoadDataset():
    return pd.read_pickle(current_dir + '/drug review dataset drugs.com/DrugsComPatient_raw.pkl')

main_data = LoadDataset()
x = main_data[['condition','review']]

condition = main_data['condition'].value_counts()
valid = condition[condition>=4000].index
data = main_data[main_data['condition'].isin(valid)]
data = data.drop("Unnamed: 0",axis=1).reset_index(drop=True)

@st.cache_data
def ProportionTopConditions(n_top=5):
    condition_counts = data["condition"].value_counts()
    valid_condition = condition_counts.head(n_top).index
    x = data[data['condition'].isin(valid_condition)]
    cond_ind=x['condition'].value_counts().index
    cond_val=x['condition'].value_counts().values

    sns.set_style(style="ticks")
    if n_top>6:
        fig= plt.figure(figsize=(7, 7))
    else:
        fig= plt.figure(figsize=(6, 6))
    plt.pie(cond_val, labels=cond_ind, autopct='%.2f%%', startangle=120, colors=sns.color_palette("deep6"))
    plt.title(f'Proportion Top {n_top} Conditions', fontsize=15)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    st.pyplot(fig)


img_mask = PIL.Image.open(current_dir +'/Med1.jpg')
img_mask = np.array(img_mask)
@st.cache_data
def PlotWordCloud(class_label=None):
    if class_label:
        data = x[x['condition'] == class_label]['review']
    fig =plt.figure(figsize=(10,8))
    wc = WordCloud(max_words=200, 
                   colormap = 'BuPu_r', mask=img_mask, 
                   background_color='black').generate(' '.join(data))
    plt.axis('off')
    plt.imshow(wc)
    plt.title(f"Word Cloud for {class_label}",fontsize=20)
    plt.tight_layout()
    fig.patch.set_alpha(0)
    st.pyplot(fig)

@st.cache_data  
def PlotMostImportantFeature(class_label,top_features=10):
    feature_names = tfidf_vecto_ubigram.get_feature_names_out()
    class_index = le.transform([class_label])[0]
    
    if isinstance(model, (PassiveAggressiveClassifier, LogisticRegression)):
        coef = model.coef_[class_index]
    elif isinstance(model, MultinomialNB):
        coef = model.feature_log_prob_[class_index]
    elif isinstance(model, (xgboost.XGBClassifier, KNeighborsClassifier)):
        st.write(f"The model '{type(model).__name__}' does not provide direct feature importances.")
        return
    else:
        st.write(f"The model '{type(model).__name__}' has no suitable attribute for extracting feature importances. Unable to plot the top feature for label {class_label}.")
        return
    
    # if len(coef.shape) > 1:
    #     coef = coef[class_index]

    top_positive_coefficients = sorted(zip(coef, feature_names), key=lambda x: x[0], reverse=True)[:top_features]
    top_negative_coefficients = sorted(zip(coef, feature_names), key=lambda x: x[0])[:top_features]
    
    # Plotting
    fig =plt.figure(figsize=(40, 40))
    plt.subplot(1, 2, 1)
    top_coefficients = [coef for coef, feat in top_positive_coefficients]
    top_features_names = [feat for coef, feat in top_positive_coefficients]
    sns.barplot(x=top_coefficients, y=top_features_names, palette="Blues_d",hue=top_features_names)
    plt.title(f"Top {top_features} Positive Features ({class_label})",fontsize=60)
    plt.ylabel("Words",fontsize=45)
    plt.xlabel("Coefficient",fontsize=45)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    
    plt.subplot(1, 2, 2)
    top_coefficients = [coef for coef, feat in top_negative_coefficients]
    top_features_names = [feat for coef, feat in top_negative_coefficients]
    sns.barplot(x=top_coefficients, y=top_features_names, palette="Reds_d",hue=top_features_names)
    plt.title(f"Top {top_features} Negative Features ({class_label})",fontsize=60)
    plt.ylabel("Words",fontsize=45)
    plt.xlabel("Coefficient",fontsize=45)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    
    plt.tight_layout()
    fig.patch.set_alpha(0)
    st.pyplot(fig)

    
stopwords = stopwords.words('english')
punct = string.punctuation
lemmatizer = WordNetLemmatizer()

def GetTag(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith(('V','N','R')):
        return tag.lower()[0]
    else:
        return None
    
def LemmatizeWord(words):
    pos = pos_tag(words)
    lemmatized_words = []
    for word, tag in pos:
        pos = GetTag(tag)
        if pos:
            lemmatized_words.append(lemmatizer.lemmatize(word,pos))
        else:
            lemmatized_words.append(word)
    return lemmatized_words

@st.cache_data
def CleanWords(text):
    # 1. Remove HTML
    review_text = BeautifulSoup(text,'html.parser').get_text()
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 3. Convert words to lower case and tokenize them
    words = word_tokenize(review_text.lower())
    # 4. Remove Punctuation
    words = [word for word in words if word not in punct]
    # 5. Remove stopwords
    words = [w for w in words if w not in stopwords]
    # 6. Lemmatize words
    words = LemmatizeWord(words)
    # 7. Join the words back into one string separated by space and return
    return " ".join(words)
    
# Load LabelEncoder
le = joblib.load(current_dir +'/label_encoder.pkl')

# Load TFIDFVectorizer
tfidf_vecto_ubigram = joblib.load(current_dir +'/Bag of Words/tfidf_vectorizer_(1,2)-gram.pkl')

model = None

@st.cache_data
def LoadModel(model_name):
    global model
    if model_name == "Passive Aggressive Classifier":
        model = joblib.load(current_dir +"/Models/pac_tfidf_(1,2)-gram_model.pkl")
    elif model_name == "Logistic Regression":
        model = joblib.load(current_dir +"/Models/lr_tfidf_(1,2)-gram_model.pkl")
    elif model_name == "Naive Bayes":
        model = joblib.load(current_dir +"/Models/mnb_tfidf_(1,2)-gram_model.pkl")  
    elif model_name == ("K-Nearest Neighbors"):
        model = joblib.load(current_dir +"/Models/knn_tfidf_(1,2)-gram_model.pkl") 
    elif model_name == "Support Vector Machine":
        model = joblib.load(current_dir +"/Models/svc_tfidf_(1,2)-gram_model.pkl")
    elif model_name == "XGBoost":
        model = joblib.load(current_dir +"/Models/xgb_tfidf_(1,2)-gram_model.pkl")
    return model

# Function to predict the condition of the patient and the top 3 drugs for the condition
@st.cache_data
def ExtractTopDrugs(label):
    data_top = data[(data['rating']>=9) & (data['usefulCount']>=100)].sort_values(by=['rating','usefulCount'],ascending=[False, False])
    data_top.head()
    drug_list = data_top[data_top['condition']==label]['drugName'][:3].tolist()
    return drug_list

@st.cache_data
def PredictCondition(text):
    global model
    text = [CleanWords(text)]
    text = tfidf_vecto_ubigram.transform(text)
    pred = model.predict(text)[0]
    return le.inverse_transform([pred])[0]

def ConfusionMatrix(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 4))
    sns.heatmap(cm,annot=True,fmt='d',cmap='coolwarm',xticklabels=le.classes_,yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {model_name}',fontsize=15)
    plt.xlabel('Predicted',fontsize=15)
    plt.ylabel('Actual',fontsize=15)
    fig.patch.set_alpha(0)
    st.pyplot(fig)
    
def Accuracy(y_true,y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    st.write(f"Accuracy: {accuracy}")

@st.cache_data
def LoadTestData():
    x_test = joblib.load(current_dir +'/Data/tfidf_test_ubigram.pkl')
    y_test = joblib.load(current_dir +'/Data/label_test.pkl')
    return x_test, y_test


st.title("Patient's Condition Classification Using Drug Reviews")

st.sidebar.title("Model")
model_name = st.sidebar.selectbox("Choose model", ["Passive Aggressive Classifier", "Logistic Regression", 
                                                   "Naive Bayes", "K-Nearest Neighbors", 
                                                   "Support Vector Machine", "XGBoost"])

# Load
model = LoadModel(model_name)
x_test, y_test = LoadTestData()
y_pred = model.predict(x_test)

if 'menu' not in st.session_state:
    st.session_state['menu'] = 'Dataset'

def set_menu(menu_name):
    st.session_state['menu'] = menu_name

st.sidebar.title("Menu")
if st.sidebar.button("Dataset"):
    set_menu('Dataset')
if st.sidebar.button("Top Proportion Conditions"):
    set_menu('Proportion')
if st.sidebar.button("Word Cloud"):
    set_menu('Word Cloud')
if st.sidebar.button("Most Important Features"):
    set_menu('Features')
if st.sidebar.button("Metrics Evaluation"):
    set_menu('Evaluation')
if st.sidebar.button("Prediction"):
    set_menu('Predict')

if st.session_state['menu'] == 'Dataset':
    st.header("Dataset",divider="rainbow")
    st.dataframe(data=data)

if st.session_state['menu'] == 'Proportion':
    st.header("Top Proportion Conditions",divider="rainbow")
    n_top = st.slider("Select Top n Conditions",min_value=3,max_value=len(data['condition'].unique()),step=1)
    ProportionTopConditions(n_top)

if st.session_state['menu'] == "Word Cloud":
    st.header("Word Cloud",divider="rainbow")
    class_label = st.selectbox("Choose Class for WordCloud", data['condition'].unique())
    PlotWordCloud(class_label)

if st.session_state['menu'] == "Features":
    st.header("Most Important Features",divider="rainbow")
    class_label = st.selectbox("Choose Class for Important Features", data['condition'].unique())
    n_top = st.slider("Select Top n Important Features",min_value=5,max_value=15,step=1)
    PlotMostImportantFeature(class_label,n_top)
        
if st.session_state['menu'] == "Evaluation":
    st.header(f"Metrics Evaluation",divider="rainbow")
    st.subheader("Confusion Matrix")
    ConfusionMatrix(model_name,y_test,y_pred)
    st.subheader(f"Accuracy: {accuracy_score(y_test,y_pred)}")
    # Accuracy(y_test,y_pred)
    st.subheader("Classification Report")
    st.text(classification_report(y_test,y_pred))

if st.session_state['menu'] == "Predict":
    st.header("Prediction",divider="rainbow")
    input_text = st.text_input(label="Insert your review: ", placeholder="Insert review...")
    if st.button("Predict"):
        prediction =PredictCondition(input_text)
        st.subheader(f"Condition: {prediction}")
        
        st.subheader("Top 3 Drug Recommendation:")
        recommendation = ExtractTopDrugs(prediction)
        for i, drug in enumerate(recommendation):
            st.write(f"{i+1}. {drug}")