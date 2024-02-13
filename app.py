import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Ensure all necessary NLTK data is downloaded
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# Load the pre-trained models
tfidf_vectorizer = load('tfidf_vectorizer.joblib')
svd_model = load('svd_model.joblib')
gb_model = load('gbmodel.joblib')

# Set of English stopwords
stop_words = set(stopwords.words('english'))


# Main app layout
def main():

    st.title('Quora Question Pairs Similarity')
    # Create tabs
    tab1, tab2 = st.tabs(["Data Visualization","Model Prediction"])


    with tab1:
        st.title('Data Visualization and EDA')

        # Load data
        og_data = pd.read_csv('questions.csv')

        # Display the first 5 rows of the data
        st.write('First 5 rows of the data')
        st.write(og_data.head())

        # Plot the distribution of question lengths
        st.write('Distribution of Target Variable: Is Duplicate')
        fig, ax = plt.subplots()
        og_data['is_duplicate'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)


        data = og_data.sample(n=10000, random_state=1)
        data = data.reset_index(drop=True)

        # Plot the distribution of question lengths
        st.write('\n\nDistribution of Question Lengths')
        fig, ax = plt.subplots()
        sns.histplot(data['question1'].apply(lambda x: len(x.split())), kde=True, ax=ax)
        sns.histplot(data['question2'].apply(lambda x: len(x.split())), kde=True, ax=ax)
        ax.set_xlabel('Question Length')
        ax.set_ylabel('Frequency')
        ax.legend(['Question 1', 'Question 2'])
        st.pyplot(fig)

        
        #Most common words
        questions = list(data['question1']) + list(data['question2'])
        len(questions)

        #find number of unique words
        unique_words = len(set(" ".join(questions).split()))
        print("Number of unique words: ", unique_words)

        #printing the most common words 
        from collections import Counter
        cnt = Counter()
        for word in questions:
            cnt.update(word.split())
        print(cnt.most_common(10))

        #plot
        st.write('Most common words')
        most_common = cnt.most_common(10)
        most_common = dict(most_common)
        plt.figure(figsize=(12, 6))
        plt.bar(most_common.keys(), most_common.values(), color='orange')
        plt.title('Most common words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.show()
        st.pyplot(plt)



      # Plot the distribution of common words
        st.write('Distribution of Common Words')
        def common_words(row):
            q1=set(map(lambda word: word.lower().strip(),row['question1'].split(" ")))
            q2=set(map(lambda word: word.lower().strip(),row['question2'].split(" ")))
            return len(q1 & q2)

        data['common_words'] = data.apply(common_words, axis=1)

        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1)
        sns.violinplot(x = 'is_duplicate', y = 'common_words', data = data)
        plt.title('Common words in questions')
        plt.show()
        st.pyplot(plt)


        #Word length differences between questions
        st.write('Word Length Differences')

        # calculate word length of questions in two columns
        data['q1_wordlen'] = data['question1'].apply(lambda x: len(str(x).split()))
        data['q2_wordlen'] = data['question2'].apply(lambda x: len(str(x).split()))

        # create a new column that has difference in word length
        data['word_difference'] = abs(data['q1_wordlen'] - data['q2_wordlen'])

        # plot to check the difference in word_difference across duplicate questions and non duplicate
        plt.figure(figsize=(12, 6))
        sns.violinplot(x = 'is_duplicate', y = 'word_difference', data = data)
        plt.title('Word length difference in questions')
        plt.show()
        st.pyplot(plt)




    def expand_contractions(text):
        # Dictionary of English contractions
        contractions_dict = {"don't": "do not", "doesn't": "does not", "didn't": "did not",
                            }
        # Regular expression for finding contractions
        contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

        def replace(match):
            return contractions_dict[match.group(0)]

        return contractions_re.sub(replace, text)

    #function to handle LaTeX expressions
    def clean_math_text(text):

        replacements = {
            # Basic operations and structures
            r'\\frac\{(.*?)\}\{(.*?)\}': r'\1 over \2',
            r'\\sqrt\{(.*?)\}': r'square root of \1',
            r'\\sum_(\{.*?\})\^(\{.*?\})': r'sum from \1 to \2',
            r'\\int_(\{.*?\})\^(\{.*?\})': r'integral from \1 to \2',
            r'\\log_(\{.*?\})\{(.*?)\}': r'log base \1 of \2',
            r'\\lim_(\{.*?\})': r'limit as \1',
            r'(\d+)\^(\{?\d+\}?)': r'\1 to the power of \2',
            r'\\infty': 'infinity',
            r'\\pm': 'plus or minus',
            # Greek letters
            r'\\alpha': 'alpha', r'\\beta': 'beta', r'\\gamma': 'gamma',
            r'\\delta': 'delta', r'\\epsilon': 'epsilon', r'\\zeta': 'zeta',
            r'\\eta': 'eta', r'\\theta': 'theta', r'\\iota': 'iota',
            r'\\kappa': 'kappa', r'\\lambda': 'lambda', r'\\mu': 'mu',
            r'\\nu': 'nu', r'\\xi': 'xi', r'\\omicron': 'omicron',
            r'\\pi': 'pi', r'\\rho': 'rho', r'\\sigma': 'sigma',
            r'\\tau': 'tau', r'\\upsilon': 'upsilon', r'\\phi': 'phi',
            r'\\chi': 'chi', r'\\psi': 'psi', r'\\omega': 'omega',
            # Trigonometric functions
            r'\\sin': 'sine', r'\\cos': 'cosine', r'\\tan': 'tangent',
            r'\\csc': 'cosecant', r'\\sec': 'secant', r'\\cot': 'cotangent',
            # Differential and partial differential
            r'\\partial': 'partial', r'\\nabla': 'nabla',
            r'\\mathrm\{d\}': 'd',  # For derivatives
            # Other mathematical symbols
            r'\\times': 'times', r'\\div': 'divided by', r'\\cdot': 'dot',
            # Additional symbols and operations
            r'\+': 'plus', r'\-': 'minus', r'\*': 'times',
            # Handling general exponentiation
            r'\\exp\{(.*?)\}': r'e to the power of \1',  # For exponential functions
            r'(\w+)\^(\w+)': r'\1 to the power of \2',  # General exponentiation
            # Handling \mathop
            r'\\mathop\{\\rm ([^}]+)\}': r'operator \1'    }
        
        # Function to apply replacements to a matched object
        def apply_replacements(match):
            # Extracting the matched text excluding the [math] tags
            math_text = match.group(1) # match.group(0) includes the whole match, so match.group(1) is the first capture group
            
            # Applying all replacements to the math_text
            for pattern, replacement in replacements.items():
                math_text = re.sub(pattern, replacement, math_text)
            
            # Return the transformed math_text
            return math_text

        # Use=ing re.sub with a function that applies the replacements for each [math] section
        # Pattern captures the content between [math] and [/math] tags
        pattern = r'\[math\](.*?)\[/math\]'
        clean_text = re.sub(pattern, apply_replacements, text)

        # Removing unnecessary braces and cleanup, applied globally to the whole text
        clean_text = re.sub(r'\{|\}', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    # Function to clean text
    def clean_text(text):
        #handling LaTex expressions
        text = clean_math_text(text)
        # Lowercase conversion
        text = text.lower()
        # Removing HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Removing URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Expanding contractions
        text = expand_contractions(text)
        # Removing special characters
        text = re.sub(r'\W', ' ', text)
        # Removing extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # removing stopwords
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text


    def expand_contractions(text):
        # Contractions map
        contractions_dict = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            }
        # Regular expression for finding contractions
        contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
        def replace(match): return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, text)

    def generate_features(clean_q1, clean_q2):
        # Calculate lengths of each question
        lengthq1 = len(clean_q1)
        lengthq2 = len(clean_q2)
        
        # Calculate word lengths of each question
        q1_wordlen = len(clean_q1.split())
        q2_wordlen = len(clean_q2.split())
        
        # Calculate common words
        q1_words = set(clean_q1.split())
        q2_words = set(clean_q2.split())
        common_words = len(q1_words.intersection(q2_words))
        
        # Calculate word difference
        word_difference = abs(q1_wordlen - q2_wordlen)
        
        return np.array([lengthq1, lengthq2, common_words, q1_wordlen, q2_wordlen, word_difference])

    def vectorize_and_reduce(question1, question2):
        # Vectorize the questions
        tfidf_q1 = tfidf_vectorizer.transform([question1])
        tfidf_q2 = tfidf_vectorizer.transform([question2])
        # Reduce dimensions
        reduced_q1 = svd_model.transform(tfidf_q1)
        reduced_q2 = svd_model.transform(tfidf_q2)
        # Calculate squared differences
        squared_differences = np.square(reduced_q1 - reduced_q2).flatten()
        return squared_differences

    def predict_duplicate_proba(question1, question2):
        # Clean the input questions
        clean_q1 = clean_text(question1)
        clean_q2 = clean_text(question2)
        
        # Generate all required features
        features = generate_features(clean_q1, clean_q2)
        
        # Vectorize and reduce the cleaned questions
        vector_features = vectorize_and_reduce(clean_q1, clean_q2)
        
        # Combine all features for prediction
        final_features_array = np.hstack((features, vector_features))
        
        # Define feature names for the DataFrame
        basic_feature_names = ['lengthq1', 'lengthq2', 'common_words', 'q1_wordlen', 'q2_wordlen', 'word_difference']
        svd_feature_names = [str(i) for i in range(vector_features.shape[0])]  # SVD feature names as '0', '1', '2', ...
        feature_names = basic_feature_names + svd_feature_names
        
        # Convert the final features array to a DataFrame with feature names
        final_features_df = pd.DataFrame([final_features_array], columns=feature_names)
        
        # Predict probabilities using the GradientBoosting model
        probas = gb_model.predict_proba(final_features_df)
        return probas



    with tab2:
        st.title('Model Prediction')

        # Set page description
        st.write('This app predicts the similarity between two questions')

        # Get user input
        question1 = st.text_input('Enter the first question')
        question2 = st.text_input('Enter the second question')

        # Predict similarity
        if st.button('Predict Similarity'):
            probas = predict_duplicate_proba(question1, question2)
            st.write('Probability of being non-duplicate: ', probas[0][0])
            st.write('Probability of being duplicate: ', probas[0][1])
            if probas[0][1] > 0.3:
                st.write('The questions are similar')
            else:
                st.write('The questions are not similar')

    

if __name__ == "__main__":
    main()
