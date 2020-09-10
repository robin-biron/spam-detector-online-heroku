# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:13:55 2020

@author: spawnersky
"""
import streamlit as st
import re
from joblib import dump, load
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import random
import markdown
from bokeh.io import output_file, show
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, dodge
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, FactorRange, HoverTool
from bokeh.plotting import figure
from bokeh.models.widgets import Panel, Tabs
import bokeh.layouts
import bokeh.models
import bokeh.plotting
from bokeh.themes import built_in_themes
import nltk
nltk.download('words')

# General formating in CSS
page_bg_img = '''
   <style>
   body {
   background-image: url("https://www.xmple.com/wallpaper/black-linear-cyan-gradient-1920x1080-c2-010506-073a47-a-120-f-14.svg");
   background-size: cover;
   color: #fff;
   }
   
   h1 {
   	color:#c4d8d6
   	;
   }
   
   h2 {
   
   color : #5a6794
   	;
   }
   
   label {
   color: #fff;
   
   ;
   }
   
   .stButton>button {
   color: #000000;
   background-color: #f6be65;
   font-size: large
 
   }
   
   .stTextArea>label {
   color: #fff;
   font-size: medium
   }
   
.stTextArea>div{
   background-color: #ddddda;
   }
   
   .stTextInput>label {
   color: #fff;
font-size: medium
   
   }
   .stTextInput>div>div{
   background-color: #ddddda;
   }
   
   
   .stSelectbox>label{
   color: #fff;
   font-size: medium
   }
   
.stSelectbox>div>div{
   background-color: #ddddda;
   }
   
   .btn-outline-secondary{
   	background-color: #f6be65;
   	
   }
   
   .btn-outline-secondary>span{
   	color: #000000
   	;
}
   .stAlert{
   background-color: #b0cac7;
 
   }
   	
   </style>
   '''
st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    
   
    # # hide streamlit menu since it doesn’t really serve any purpose to the user, so there’s no reason the user should see it
    # hide_streamlit_style = """
    # <style>
    # #MainMenu {visibility: hidden;}
    # footer {visibility: hidden;}
    # </style>   
    # """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True)   
    
    # Activate sidebar functions
    _set_block_container_style(sidebar_settings())
    st.sidebar.header("Navigation menu")
    
    # Activate navigation within web app pages 
    navigate = st.sidebar.radio("Select a page", ('Home', 'Our Methodology', 'About','Feedback'))
    if navigate=='Home':
        home()
    elif navigate=='Our Methodology':
        methodology()
    elif navigate=='About':
        about()
    elif navigate=='Feedback':
        feedback()
            
## Page 'Home' ##
def home():
    # General formating and layout
   
    # Copywritting, storystelling and formatting
    st.markdown(
            """
    # The #1 Spam Detector Online
    Email Spam Detection Using Python & Machine Learning
    
    """
        )
     
    st.markdown(
            """
    ## Zero Risk does not exist
    """)
    
    text="""
    Spear-phishing attacks playing on people's fears related to Coronavirus 
    increased by **667%** between February and June 2020
    """
    
    image = Image.open('covid.jpg')
    st.image(image, use_column_width=True)
    
    st.markdown(""" 
                There are many reasons in today's environment to be wary of email that seems in any way suspicious.
                Some email messages might be phishing scams; some might contain viruses and other malicious software.
                """)
                
    st.markdown("""
               According to ANSII (National Information Systems Security Agency), phishing aims at making the recipient of an apparently legitimate email send his bank data or login credentials to financial services, 
                in order to steal money from him. In addition to phishing, spam emails can also take the form of advertising or a scam also known as "Nigerian Scam.
                """)
                
    st.markdown("""  
                Coronavirus spear phishing email attacks detected by Barracuda Networks increased from 137 in January 2020 to more than 9,116 in the first 23 days of March 2020.
                """)
    st.markdown("""
                Source: [blog.barracuda](https://blog.barracuda.com/2020/03/26/threat-spotlight-coronavirus-related-phishing/)
                """)
    
    st.markdown(
            """
    ## Unfortunately, current spam filters are not 100% efficient
    Spam detectors included in free email services (Gmail, Outloook, AOL, GMX, etc.) are not 100% efficient.
    Why ? Because they only focus on the header analysis. That's a mistake. It's been a decade experts agree that 
    NLP in machine learning is one of the most powerful tool to detect suspicious emails. 
    
    """
        )
    
    st.markdown("""
                That's why we focus here on both body- and subject content. Also we include the issuer address and the content type in our analysis since they 
    provide significant information in email analysis.
    """)
    

    
    st.markdown("""
            ### Almost there !
            You just need to follow the following steps and you 
            will get the results within a few seconds. Let's' get started !"""
    )
    
    st.info("Since we trained our machine learning model on english emails, the spam detector only works with english content.")
    
    
    # Beginning of interractions with user
    
    X_content=st.text_area("1) Paste body content:","Enter a message...")
    X_content=str(X_content)
    X_subject=st.text_area("2) Paste subject content:","Enter a message...")
    X_subject=str(X_subject)
    X_from=st.text_input("3) Paste issuer address (FROM):","Enter an address...")
    X_from=str(X_from)
    
    if X_content=='Enter a message...':
        X_content=None
    if X_subject=='Enter a message...':
        X_subject=None
    if X_from=='Enter an address...':
        X_from=None
    
    
    options = st.selectbox('4) Select Content-Type:',
                                  ('plaintext', 'html'))
    
    if options=='plaintext':
        X_type_html=0
        X_type_plain=1
    else:
        X_type_html=1
        X_type_plain=0
                
    
    # Start analysis only if the user ckecks the 'Analyze' button and filled in all text areas
    
    if  st.button('Analyze'):
        try:
            #### Creation of numeric features ####
            #nb of words
            count_body = len(re.findall(r'\w+', X_content))
            count_subject= len(re.findall(r'\w+', X_subject))
            
            #list of words
            words_body=re.findall(r'\w+', X_content)
            words_subject=re.findall(r'\w+', X_subject)
            
            #length of message and subject
            body_len=len(X_content)
            subject_len=len(X_subject)
            
            #nb of upper words
            for word in words_body:
                if len(word)>1:
                    upper_words_body_nb = sum(1 for c in words_body if c.isupper())
            
            for word in words_subject:
                if len(word)>1:
                    upper_words_subject_nb = sum(1 for c in words_subject if c.isupper())
            
            #nb of special characters
            liste_ponct = ['€','!','"','#','$','%','&','(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[',']','^','_','`','{','|','}','~']
            
            temp = sum(1 for c in X_content if c in liste_ponct)
            if temp > 0:
                symb_body_nb=temp
            else:
                symb_body_nb=0
                 
            temp = sum(1 for c in X_subject if c in liste_ponct)
            if temp > 0:
                symb_subject_nb=temp
            else:
                symb_subject_nb=0
            
            #nb of http url
            r=re.compile(r"http?://[a-zA-z0-9./-]+")
            
            http_body_nb = len(re.findall(r, X_content))
            
            # space nb + ratio (nb_space/message_len)
            space_body_nb=0
            space_body = sum(1 for space in X_content if space in ' ')
            if space_body > 0 :
                space_body_nb=space_body
                
            for word in words_body:
                if len(word)>1:
                        space_body_ratio=space_body_nb/body_len
            
            #presence of a figure in issuer address (i.e From field)
            if '0' in X_from:
                n = 1
            elif '1' in X_from:
                n=1
            elif '2' in X_from:
                n=1
            elif '3' in X_from:
                n=1
            elif '4' in X_from:
                n=1
            elif '5' in X_from:
                n=1
            elif '6' in X_from:
                n=1
            elif '7' in X_from:
                n=1
            elif '8' in X_from:
                n=1
            elif '9' in X_from:
                n=1
            else:
                n=0
            figure_issuer_address=n
            
            #nb of words not in dictionary (english) (we refer to a dictionnary composed of 236736 words)
            
            from nltk.corpus import words
            word_list_en = set(words.words())
            
            count_body_dico=0
            for word in words_body:
                if word not in word_list_en:
                    count_body_dico+=1
            not_dico_body=count_body_dico
            
            count_subject_dico=0
            for word in words_subject:
                if word not in word_list_en:
                    count_subject_dico+=1
            not_dico_subject=count_subject_dico
            
            #nb of sexual adults words
            adult_words=['sex','sexual','erotic','porn','xxx','adult','sexy']
            
            count_adult_body=0
            for word in words_body:
                if word in adult_words:
                    count_adult_body+=1
            adult_words_body=count_adult_body
            
            count_adult_subject=0
            for word in words_subject:
                if word in adult_words:
                    count_adult_subject+=1
            adult_words_subject=count_adult_subject
            
            # creation of dataframe
             
            s1=pd.Series({'words_body_nb':count_body, 'body_len':body_len,'upper_words_body_nb':upper_words_body_nb,
                   'body_content':X_content,'http_body_nb':http_body_nb, 'symb_body_nb':symb_body_nb,
                     'text/html':X_type_html,'text/plain':X_type_plain,
                   'upper_words_subject_nb':upper_words_subject_nb,'symb_subject_nb':symb_subject_nb,
                  'words_subject_nb':count_subject,'subject_len':subject_len, 
                   'subject_len':subject_len,'adult_words_body':adult_words_body,
                   'adult_words_subject':adult_words_subject,'not_dico_body':not_dico_body,
                   'not_dico_subject':not_dico_subject,'figure_issuer_address':figure_issuer_address,
                    'space_body_nb':space_body_nb,'space_body_ratio':space_body_ratio,
                   'subject_content':X_subject
                    })
        
            df1=pd.DataFrame([s1])
            df_body=df1.copy()
            df_subject=df1.copy()
       
            # ***** Prediction of spam/ham *****
            
            # Loading of joblib objects already fitted with thousands of data
            model =load("rf.joblib")
            vectorizer_body = load("vectorizer_body.joblib")
            vectorizer_subject = load("vectorizer_subject.joblib")
            
            # Tokenization of text contents
            X_content_nlp = vectorizer_body.transform([X_content])
            X_subject_nlp = vectorizer_subject.transform([X_subject])
            X_nlp_test3 = hstack((X_content_nlp,X_subject_nlp, df1.drop(columns=['body_content','subject_content']).values))
            
            # Start prediction 
            pred=model.predict(X_nlp_test3)
            
            if pred==1:
                st.markdown("""## **It might be considered as spam**
                            """)
                style_spam = '''
                <style>
                    
                    h2>strong {
                        
                        color : #b74355
                    	;
                    }
                    
                    </style>
                    '''
                st.markdown(style_spam, unsafe_allow_html=True)
    
            else:
                st.markdown("""## **It might be considered as ham**
                            """)
                style_spam = '''
                <style>
                    
                    h2>strong {
                        
                        color : #99d594
                    	;
                    }
                    
                    </style>
                    '''
                st.markdown(style_spam, unsafe_allow_html=True)
            
            tabs = bokeh.models.Tabs(
        tabs=[
            body_analysis_panel(df_body),subject_analysis_panel(df_subject)
        ]
    )
            st.markdown("""
            ### Want To Learn More About The Results?
            """)
            st.bokeh_chart(tabs)
    
        # Start the analysis only if the user filled in all text areas
        except TypeError:
            st.write("First you need to fill in the text areas")

## Page 'Project' ##

def methodology():

    st.title('Building a Spam Detector from Scratch Using Machine Learning')
    st.subheader('Robin Biron, Félix Peyré, Alexis Teskrat ')
    st.write('Sept 10, 2020 · 20 min read :star:')
    

    image = Image.open('hacker.jpeg')
    st.image(image, use_column_width=True)

    st.info("This page is currently under maintenance")
    # ##### intro #######
    # df = pd.read_csv(r'df_spam_ham.csv')

    # st.header('Introduction')
    # intro = """According to ANSII (French Information Systems Security Agency) :
    # **phishing** aims at making the recipient of an apparently legitimate email
    # send his bank data or login credentials to financial services, in order to steal
    # money from him. In addition to phishing, spam emails can also take the form of
    # advertising or a scam also known as "Nigerian Scam". It is acknowledged that 94% of
    # cyberthreats start with an email (source Vade Secure), 67% of ransonware attacks
    # start with a phishing or spam email (Statistica) all this resulting in a loss
    # of $ 1.77 billion for companies in 2019."""
    # st.write(intro)
    # st.write("""We therefore understand that it is necessary
    # to create tools capable of detecting such frauds. In machine learning we call it a **binary
    # classification problem**, the goal being to say whether or not an email is malicious in
    # view of its content.""")
    
    # ##### Method #######
    # st.header('Methodology')
    # st.subheader('Our Dataset')
    # st.write("""The Apache SpamAssassin project [SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/)
    # provides open source mails in 'mailbox file' format.
    # We have worked on a dataset of 3000 mails written in English and divided into two folders spam and ham (healthy e-mail).
    # Each email contains headers such as sender address, date, MIME version etc.""")
    
    # st.write("Our dataset is unbalanced, which is quite remarkable when we pay attention to the following pie chart:")
    # image4 = Image.open('proportion-ham-spam.jpg')
    # st.image(image4, width=500)
    # st.write("""Indeed hams represent 83.69% of all of our emails against 16.31% for spam
    # The header of each email contains a lot of information when it is filled (more than 200 features),
    # but when we look at the number of missing values in the headers of our mails, this is what we get:
    # """)
    
    # non_bin = ['From','Subject','Date','upper_words_subject_nb','symb_subject_nb', 'words_subject_nb',
    # 	'not_eng_body','not_eng_subject','nb_in_adress_exp', 'words_body_nb', 'symb_body_nb',
    # 	'message_len','multipart/related','multipart/mixed','multipart/report','multipart/signed',
    # 	'text/html','text/plain','upper_words_body_nb','content','http_body_nb','Message-ID',
    # 	'subject_len','adults_words_body','target','space_body_nb', 'attachment', 'hour_int_(7, 23]',
    # 	'space_body_ratio','hour','day','weekday','fortnight_(15, 31]','weekday_int_(4, 6]']
    # df_bin = df.drop(columns = non_bin)
    
    # res = pd.DataFrame(index = [0,1])
    # for var in df_bin:
    # 	res = res.join(df_bin[var].value_counts(normalize=True))
    
    # res.drop(index= 1,inplace=True)
    # res = pd.DataFrame(np.array(res).reshape(df_bin.shape[1],1),index=res.columns,columns=["Null_ratio"])
    
    # p = figure(title="Histogram of the ratio of missing values in header features",plot_width = 700, plot_height = 400,x_range = (0,1),x_axis_label= "Percentage of missing values", y_axis_label="Number of features")
    
    # hist, edges = np.histogram(res, bins = 10)
    # hist_df = pd.DataFrame({"Null_ratio": hist,"left": edges[:-1],"right": edges[1:]})
    # hist_df["interval"] = ["%s to %s" %(round(left,2),round(right,2)) for left, right in zip(hist_df["left"], hist_df["right"])]
    
    # source = ColumnDataSource(hist_df)
    # p.quad(bottom = 0, top = "Null_ratio",left = "left", right = "right", source = source, fill_color = 'blue', line_color = "black", fill_alpha = 0.7, hover_fill_alpha = 1.0)
    
    # hover = HoverTool(tooltips = [('Interval', '@interval'),('Count', str("@" + "Null_ratio"))])
    # p.add_tools(hover)
    
    # st.bokeh_chart(p)
    
    
    # st.write("""We notice that 199 header features have more than 90% missing values in our dataset.
    # In order to manage the missing values, a binarization of these features has been carried out according to whether they have been entered or not: 0 (absence of information), 1 (presence of information).""")
    
    # st.subheader('Features and DataFrame')
    # st.write("""A DataFrame containing all the emails has been created with the body of the email, its subject and also the information on the presence or absence of information in the header as features.
    # Other features have also been created based on the body of the email, the subject or the address of the sender.
    # For example, we tried to enter the number of special characters, the number of words in upper case, the number of url links, etc.""")
    # st.write("The first 5 rows of the DataFrame are visible below :")
    
    # st.write(df.head())
    
    # st.write("To get an idea of the content of the mails, we display the wordclouds for both types of mail.")
    # st.write("Most frequent words in **spams** :")
    # image9 = Image.open('wc-spam.png') 
    # st.image(image9, width=300)
    # st.write("Most frequent words in **hams** :")
    # image10 = Image.open('wc-ham.png') 
    # st.image(image10, width=300)
     
    
    # ############## Machine learning #######
    # st.header('Machine learning')
    # st.write("Two machine learning models have been conducted and compared to address this classification problem.")
    # st.subheader('Model A')
    
    # st.write("""** Model A ** offers a succession of two classification tests. First of all the sets of
    # test and validation are separated like this : from one side the body of the email on which the NLP algorithm will be processed
    # (Natural Language Processing) and on the other hand the set of numeric features.
    # The body of the email is tokenized so as to perform a first classification test by logistic regression.
    # At the end of this test, the probabilities of belonging to each label are calculated and added as
    # new numeric feature. The new dataframe obtained containing the numeric features is standardized
    # and a reduction of its dimensions by PCA is done before performing a second classification test by SVM which this time will assign the final predictive labels.""")
    # image2 = Image.open('method_VA.png')
    # st.image(image2, use_column_width=True)
    
    # ############## CODES MODEL A ###################
    # st.write("The original dataset is separated into a training set containing features (X_train) and labels (y_train) and a test set (X_test and y_test).")
    # with st.echo():
    #     from sklearn.model_selection import train_test_split
    #     df.dropna(inplace=True)
    #     data = df.drop('target', axis=1)
    #     target = df.target
    #     X_train, X_test, y_train, y_test = train_test_split(data, target,
    #                                                         test_size=0.2)
    #     pd.set_option('mode.chained_assignment', None)
    # st.write("The training and test sets are separated into two subsets: a first containing the numeric features (X_train_num / X_test_num and a second set containing the text of the email body X_train_nlp / X_test_nlp).")
    # with st.echo():
    #     X_train_num = X_train.select_dtypes(['int64', 'float64', 'uint8'])
    #     X_test_num = X_test.select_dtypes(['int64', 'float64', 'uint8'])
    #     X_train_nlp = X_train.content
    #     X_test_nlp = X_test.content
    # st.write("The _CountVectorizer_ function is used to tokenize the set containing the text type features.")
    # with st.echo():
    #     from sklearn.feature_extraction.text import CountVectorizer
    #     vectorizer = CountVectorizer()
    #     X_train_nlp = vectorizer.fit_transform(X_train_nlp).todense()
    #     X_test_nlp = vectorizer.transform(X_test_nlp).todense()
    # st.write("""First of all we perform a logistic regression classification test. The _class_weight = 'balanced'_ argument improves the performance of the algorithm on an unbalanced dataset.
    # The algorithm is trained on the training set using the _fit_ function.""")
    # with st.echo():
    #     from sklearn.linear_model import LogisticRegression
    #     LR = LogisticRegression(max_iter=200, solver='liblinear',
    #                             penalty='l1', C=1, class_weight='balanced')
    #     LR.fit(X_train_nlp, y_train)
        
    # st.write("A first prediction of the labels is performed on the test set, thus making it possible to calculate a first precision score and to estimate the probabilities of belonging to each label (results visible in the ** Results ** part).")
    # with st.echo():
    #     y_pred_nlp = LR.predict(X_test_nlp)
    #     score1 = LR.score(X_test_nlp, y_test)
    #     proba_train_nlp = LR.predict_proba(X_train_nlp)
    #     proba_test_nlp = LR.predict_proba(X_test_nlp)
    # st.write("The previously calculated probabilities are added as new features to the set containing the numeric features.")
    # with st.echo():
    #     X_train_num['nlp_proba'] = proba_train_nlp[:,0]
    #     X_test_num['nlp_proba'] = proba_test_nlp[:,0]
        
    # st.write("The numeric features are not all of the same order of magnitude, so we carry out a normalization via the _MinMaxScaler_ function.")
    # with st.echo():
    #     from sklearn.preprocessing import MinMaxScaler
    #     scalerM = MinMaxScaler()
    #     X_train_scaled_M = scalerM.fit_transform(X_train_num)
    #     X_test_scaled_M = scalerM.transform(X_test_num)
        
    # st.write("We have a lot of numeric features (> 250), a reduction of dimension by _PCA_ allows to reduce considerably the number of features.")
    # with st.echo():
    #     from sklearn.decomposition import PCA
    #     pca_M = PCA(0.9)
    #     X_train_pca_M = pca_M.fit_transform(X_train_scaled_M)
    #     X_test_pca_M = pca_M.transform(X_test_scaled_M)
        
    # st.write("A second classification algorithm, this time _SVM_, is performed on the training set containing the reduced numeric features.")
    # with st.echo():
    #     from sklearn.svm import SVC
    #     svm = SVC(gamma='scale', class_weight='balanced', kernel='poly')
    #     svm.fit(X_train_pca_M, y_train)
    # st.write("The predictions of the final labels and the calculation of the final precision score can then be carried out on the test set (results visible in the ** Results ** part).")
    # with st.echo():
    #     y_pred_svm = svm.predict(X_test_pca_M)
    #     score2 =  svm.score(X_test_pca_M, y_test)
    
    # ######################################
        
    # st.subheader('Model B')    
    # st.write("""** Model B ** allows simultaneous processing of text features and numeric  
    # features through the creation of a sparse matrix after tokenization of text  
    # features. A unique Random Forest classification test will be carried out on this 
    # sparse matrix, which will allow predictive labels to be assigned. """)
    # image3 = Image.open('method_VB.png')
    # st.image(image3, use_column_width=True)
    
    # ############### CODES MODEL B ####################
    
    # st.write("First, we will use the same training and testing training set as previously.")
    # with st.echo():
    #     df_train = X_train.copy()
    #     df_test = X_test.copy()
    
    # st.write("We remove the features related to the header and we keep the body and subject of the emails as well as the features we have created (number of words in capital letters, number of words in a foreign language, the presence of special characters etc.)")
    # with st.echo():
    #     df_train.drop(columns=['hour', 'weekday', 'day', 'multipart/mixed', 
    #                            'multipart/related', 'multipart/report',
    #                            'multipart/signed', 'attachment', 'hour_int_(7, 23]',
    #                            'fortnight_(15, 31]', 'weekday_int_(4, 6]'],
    #                            inplace=True)
    #     df_test.drop(columns=['hour', 'weekday', 'day', 'multipart/mixed', 
    #                           'multipart/related', 'multipart/report',
    #                           'multipart/signed', 'attachment', 'hour_int_(7, 23]',
    #                           'fortnight_(15, 31]', 'weekday_int_(4, 6]'],
    #                           inplace=True)  
        
    #     df_streamlit_train = df_train.iloc[:,233:]
    #     df_streamlit_test = df_test.iloc[:,233:]
        
    #     df_streamlit_test['subject'] = df_test.Subject
    #     df_streamlit_train['subject'] = df_train.Subject
        
        
    # st.write("""The content of the email and its subject must be vectorized to be able to be used in an NLP algorithm.
    # We take care to remove the common words from the English language, then we create a vectorizer object for the body of the email and another for the subject.""")
    # with st.echo():    
    #     from nltk.corpus import stopwords
    #     from sklearn.feature_extraction.text import TfidfVectorizer
        
    #     stopWords = set(stopwords.words('english'))
    
    #     vectorizer_content = TfidfVectorizer(min_df=10, stop_words=stopWords)
    #     vectorizer_subject = TfidfVectorizer(min_df=10, stop_words=stopWords)
    
    # st.write("""We adjust our vectorizers on our respective training sets (the email body and the email subject)
    # Then we transform our training and test texts with these vectorizers.""")
    # with st.echo():
    #     X_train_content = vectorizer_content.fit_transform(df_streamlit_train.content)
    #     X_test_content = vectorizer_content.transform(df_streamlit_test.content)
    
    #     X_train_subject = vectorizer_subject.fit_transform(df_streamlit_train.subject)
    #     X_test_subject = vectorizer_subject.transform(df_streamlit_test.subject)
    
    # st.write("Finally, using the _hstack_ function, we concatenate the vecotrized text and object with the numeric features we created.")
    # with st.echo():
    #     from scipy.sparse import hstack
        
    #     X_train_nlp = hstack((X_train_content, X_train_subject, df_streamlit_train\
    #                           .drop(columns=['content','subject']).values))
    #     X_test_nlp = hstack((X_test_content,X_test_subject, df_streamlit_test\
    #                          .drop(columns=['content','subject']).values))
    
    # st.write("We apply a Random Forest algorithm to address our classification problem (results visible in the ** Results ** section)")
    # with st.echo():
    #     from sklearn.ensemble import RandomForestClassifier
        
    #     rf = RandomForestClassifier()
    #     rf.fit(X_train_nlp, y_train)
    #     pred = rf.predict(X_test_nlp)
       
        
    # ######## RESULTS #############
        
        
    # st.header('Results')
    # st.write("The confusion matrix remains one of the best way to assess a classification test, to which we add a calculation of the precision score and the recall score. Indeed the false positive rate is the main stake of spam filtering AI.")
    
    # st.subheader('Model A')
    
    # image6 = Image.open('score_VA_1.png')
    # st.image(image6, use_column_width=True)
    # st.write("The first model A classification test based only on text analysis already has a high accuracy score (> 97%), however the confusion matrix shows that 7 hams were classified as spam. These errors contribute to lower the F1 score. In the context of an unbalanced dataset, the F1 score is the indicator most representative of the reliability of the model and its ability to avoid false positives.")
    # image7 = Image.open('score_VA_2.png')
    # st.image(image7, use_column_width=True)
    # st.write("The second cassification test increases the precision score (> 0.99). The confusion matrix shows that the model failed once by classifying spam as ham. There are no false positives because all of the hams have been successfully processed by the model, which enables the model to get a F1 score of 1.")
    
    # st.subheader('Model B')
    
    # image8 = Image.open('score_VB.png')
    # st.image(image8, use_column_width=True)
    # st.write("The model B allows to process all types of predictions in a single step thanks to its sparse matrix. It provides excellent accuracy (> 99%) as well as an F1 score of 1 for the processing of false positives.")
    
    # ###### FURTHER DISCUSSION #######
    
    # st.header('Further Discussion')
    
    # st.subheader('Dataset review')
    
    # st.write("Let's take a look at our dataset")
    # st.write("""The latter includes a set of about 3000 emails with a spam / ham ratio of 20%.
    # The number of mails is moderate and the distribution between healthy mail and fraudulent mail is unbalanced.
    # With a higher number of mails (especially spam) in the dataset we would have even more robust algorithms.")
    # st.write("In addition, these emails come from a single mailbox and date from August, September and October 2002.
    # We are aware that these constraints brought by this dataset most certainly add a bias to our results.
    # Fraud methods have evolved over the past 20 years. It could be interesting to train our models on larger and better balanced datasets to gain in robustness.""")
    
    # st.subheader('Acknowledgments')
    
    # st.write("""We would f""irst like to thank the Datascientest team for its responsiveness and expertise, and in particular
    # our cohort leader Thomas Boehler who guided us throughout our work.
    # At the same time, we would like to thank The Apache Software Foundation who provided the dataset for free online
    # within the SpamAssassin project.""")
    # st.write("Thanks for reading us :smiley:")




 
    
    


## Page 'About' ##

def about():

    st.title("About Us")

    # image = Image.open('img3.jpg')
    # st.image(image,use_column_width=True)
    
    st.info(
        "This project was led by three data scientists : [Alexis Teskrat](https://www.linkedin.com/in/alexis-teskrat-a879a7195/),"
        "[Félix Peyré](https://www.linkedin.com/in/f%C3%A9lix-peyre-8997b940/)"
        " and [Robin Biron](https://www.linkedin.com/in/robin-biron-48448282/).\n\n"
        "This app shows available data in [Kaggle]"
        "(https://www.kaggle.com/veleon/ham-and-spam-dataset) dataset."
        " The latter initially comes from [SpamAssassin Project](https://spamassassin.apache.org/old/publiccorpus/).\n\n"
        "It is maintained by [Robin](https://www.linkedin.com/in/robin-biron-48448282/). \n\n"
        "Check the code at https://github.com/spawnersky/code"
    )
# https://spamassassin.apache.org/old/publiccorpus/


def feedback():
    st.title("Feedback")
    
    st.info(
        "Please feel free to contact us if you have any questions or remarks that could improve the overall website.\n\n"
        "Send us an email at rbn.biron@gmail.com"
    )
# Sidebar configuration functions

def sidebar_settings():
    """Add selection section for setting setting the max-width and padding
    of the main block container"""
    st.sidebar.header("Improve visual comfort")
    max_width_100_percent = st.sidebar.checkbox("default value size", False)
    if not max_width_100_percent:
        max_width = st.sidebar.slider("Adjust to the size you want", 100, 700, 700, 50)
    else:
        max_width = 1200
        
    return max_width
        
def _set_block_container_style(
    max_width: int = 1200, max_width_100_percent: bool = False):
    if max_width_100_percent:
        max_width_str = f"max-width: 95%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
</style>
""",
        unsafe_allow_html=True,
    )  
# Images 

def _markdown(text):
    return bokeh.models.widgets.markups.Div(
        text=markdown.markdown(text), sizing_mode="stretch_width"
    )


# Pannel functions

def body_analysis_panel(param):
    text = """
## Email Body Deep Dive Analysis
The following bar chart compares the spam median value* and your email value for different metrics.
The chosen metrics are those with the highest correlation coefficient in the correlation matrix (heatmap) : 
    
- words_body_nb : number of words in the email body
- body_len : length of email body
- upper_words_body_nb : number of upper words in the email body
- symb_body_nb : number of special characters in the email body
- adults_words_body : number of words refering to sexually explicit content
- not_dico_body : number of words not in the english dictionary (we trained our model on a dictionnary composed of 236,736 words)
- space_body_nb : number of spaces in the email body (spam email tend to have higher space ratio: number of spaces divided by the length of message)

*analysis is based on 500 spams
"""
    # import the original dataframe made of 3000 emails and calculate the median of values of different created features only for the 500 spams
    df = pd.read_csv('df_spam_ham.csv',index_col=0)
    
    df.drop(columns=['hour','weekday','day','multipart/mixed','multipart/related',
                     'multipart/report','multipart/signed','attachment',
                     'hour_int_(7, 23]','fortnight_(15, 31]','weekday_int_(4, 6]'],
            inplace=True)
    
    subject=df.Subject
    df_streamlit=df.iloc[:,233:]
    df_streamlit['subject']=subject
    
    df_body=df_streamlit[df_streamlit.target==1]
    
    # Rename columns to align on other dataframe format
    
    df_body.rename(columns={"message_len": "body_len", "content": "body_content",
                                 "not_eng_body":"not_dico_body","not_eng_subject":"not_dico_subject",
                                "nb_in_adress_exp":"figure_issuer_address",
                                "adults_words_subject":"adult_words_subject",
                                "adults_words_body":"adult_words_body",
                                "subject":"subject_content"},inplace=True)
    
    # Drop features that are not needed here from original dataframe
    df_body.drop(columns=['target','body_content','text/plain','text/html',
                     'figure_issuer_address','subject_content','http_body_nb',
                     'upper_words_subject_nb','symb_subject_nb','words_subject_nb',
                     'subject_len','adult_words_subject','not_dico_subject','space_body_ratio',
                     ]
            ,inplace=True)
    # Select median values for each category
    stats=df_body.describe()
    df_stat=pd.DataFrame([stats.iloc[5]])


    
    # Drop features that are not needed here from new dataframe
    param.drop(columns=['body_content','text/plain','text/html',
                     'figure_issuer_address','subject_content','http_body_nb',
                     'upper_words_subject_nb','symb_subject_nb','words_subject_nb',
                     'subject_len','adult_words_subject','not_dico_subject','space_body_ratio',
                     ]
            ,inplace=True)
    

    # Build Stacked histogram with Hover Tools
  
    col = df_stat.columns.tolist()
    liste1=df_stat.values.tolist()[0]
    liste_param=param.values.tolist()[0]
    
    email = ["Your email", "Spam Median Value*"]
    # colors = ["#7ae0b3", "#b74355"]
    
    data = {'col' : col,
            'Your email'   : liste_param,
            'Spam Median Value*'   : liste1}

    source = ColumnDataSource(data=data)

    p = figure(x_range=col,plot_height=400,plot_width=300, title="Email Body Deep Dive Analysis",
               toolbar_location=None,tools="")
      
    p.vbar(x=dodge('col', -0.25, range=p.x_range), top='Your email', width=0.2, source=source,
           color="#f6be65", legend_label="Your email")
    
    p.vbar(x=dodge('col',  0.0,  range=p.x_range), top='Spam Median Value*', width=0.2, source=source,
           color="#c4d8d6", legend_label="Spam Median Value*")
    

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"
    
    
    
    
    # p = figure(x_range=col, plot_height=400,plot_width=300, title="Email Body Deep Dive Analysis",
    #             toolbar_location=None, tools="hover", tooltips=" @col: @$name")
    
    # p.vbar_stack(email, x='col', width=0.6, color=colors, source=data,
    #               legend_label=email)
    
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = "#030d10"
    p.axis.minor_tick_line_color = "#030d10"
    p.outline_line_color = "#030d10"
    p.legend.location = "top_right"
    p.legend.orientation = "horizontal"
    p.xaxis.major_label_orientation = 1
    p.background_fill_color = "#030d10"
    # p.legend.background_fill_color = "#030d10"
    p.legend.click_policy="hide"
    # p.border_fill_color = "whitesmoke"


    
    
    layout = bokeh.layouts.column(
        _markdown(text), p, sizing_mode="stretch_width"
    )
    
    return bokeh.models.Panel(child=layout, title="Body")

def subject_analysis_panel(param):
    text = """
## Email Subjet Deep Dive Analysis

The following bar chart compares the spam median value* and your email value for different metrics.
The chosen metrics are those with the highest correlation coefficient in the correlation matrix (heatmap) : 

- words_subject_nb : number of words in the email subject
- subject_len : length of email subject
- upper_words_subject_nb : number of upper words in the email subject
- symb_subject_nb : number of special characters in the email subject
- adults_words_subject : number of words refering to sexually explicit content in the email body
- not_dico_subject : number of words not in the english dictionary (we trained our model on a dictionnary composed of 236,736 words)
- space_subject_nb : number of spaces in the email subject (spam email tend to have higher space ratio: number of spaces divided by the length of message)

"""

    # import the original dataframe made of 3000 emails and calculate the median of values of different created features only for the 500 spams
    df = pd.read_csv('df_spam_ham.csv',index_col=0)
    
    df.drop(columns=['hour','weekday','day','multipart/mixed','multipart/related',
                     'multipart/report','multipart/signed','attachment',
                     'hour_int_(7, 23]','fortnight_(15, 31]','weekday_int_(4, 6]'],
            inplace=True)
    
    subject=df.Subject
    df_streamlit=df.iloc[:,233:]
    df_streamlit['subject']=subject
    
    df_subject=df_streamlit[df_streamlit.target==1]
    
    # Rename columns to align on other dataframe format
    df_subject.rename(columns={"message_len": "body_len", "content": "body_content",
                                 "not_eng_body":"not_dico_body","not_eng_subject":"not_dico_subject",
                                "nb_in_adress_exp":"figure_issuer_address",
                                "adults_words_subject":"adult_words_subject",
                                "adults_words_body":"adult_words_body",
                                "subject":"subject_content"},inplace=True)
    
    # Drop features that are not needed here from original dataframe
    df_subject.drop(columns=['target','body_content','text/plain','text/html',
                     'figure_issuer_address','subject_content','http_body_nb',
                     'upper_words_body_nb','symb_body_nb','words_body_nb',
                     'body_len','not_dico_body','space_body_ratio',
                     'adult_words_body','space_body_nb']
            ,inplace=True)
    # Select median values for each category
    stats=df_subject.describe()
    df_stat=pd.DataFrame([stats.iloc[5]])


    
    # Drop features that are not needed here from new dataframe
    param.drop(columns=['body_content','text/plain','text/html',
                     'figure_issuer_address','subject_content','http_body_nb',
                     'upper_words_body_nb','symb_body_nb','words_body_nb',
                     'body_len','not_dico_body','space_body_ratio',
                     'adult_words_body','space_body_nb']
            ,inplace=True)

    # Build Stacked histogram with Hover Tools
    col = df_stat.columns.tolist()
    liste1=df_stat.values.tolist()[0]
    liste_param=param.values.tolist()[0]
    
    email = ["Your email", "Spam Median Value"]
    # colors = ["#7ae0b3", "#b74355"]
    
    data = {'col' : col,
            'Your email'   : liste_param,
            'Spam Median Value*'   : liste1}

    source = ColumnDataSource(data=data)

    p = figure(x_range=col,plot_height=400,plot_width=300, title="Email Subject Deep Dive Analysis",
               toolbar_location=None,tools="")
    
     
    p.vbar(x=dodge('col', -0.25, range=p.x_range), top='Your email', width=0.2, source=source,
           color="#f6be65", legend_label="Your email")
    
    p.vbar(x=dodge('col',  0.0,  range=p.x_range), top='Spam Median Value*', width=0.2, source=source,
           color="#c4d8d6", legend_label="Spam Median Value*")
    
    

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"
    
    
    
    
    # p = figure(x_range=col, plot_height=400,plot_width=300, title="Email Body Deep Dive Analysis",
    #             toolbar_location=None, tools="hover", tooltips=" @col: @$name")
    
    # p.vbar_stack(email, x='col', width=0.6, color=colors, source=data,
    #               legend_label=email)
    
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = "#030d10"
    p.axis.minor_tick_line_color = "#030d10"
    p.outline_line_color = "#030d10"
    p.legend.location = "top_right"
    p.legend.orientation = "horizontal"
    p.xaxis.major_label_orientation = 1
    p.background_fill_color = "#030d10"
    # p.legend.background_fill_color = "#030d10"
    p.legend.click_policy="hide"
    # p.border_fill_color = "whitesmoke"
    
    text1="""
*analysis is based on 500 spams
"""
    
    
    layout = bokeh.layouts.column(
        _markdown(text), p, _markdown(text1),sizing_mode="stretch_width"
    )
    
    return bokeh.models.Panel(child=layout, title="Subject")


main()



                                        