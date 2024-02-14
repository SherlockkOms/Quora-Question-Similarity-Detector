import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        st.markdown('---')
        # Load data
        og_data = pd.read_csv('questions.csv')
        # Display the first 5 rows of the data
        st.markdown("## First 5 rows of the data")
        st.markdown("This table shows the first 5 rows of the data")
        st.write(og_data.head())

        # Plot the distribution of question lengths using plotly
        st.markdown('## Distribution of Target Variable')
        fig = px.histogram(
        og_data, 
        x='is_duplicate', 
        title='Distribution Plot',
        labels={'is_duplicate': 'Is Duplicate'},  # Rename axis labels for clarity
        text_auto=True,  # Automatically add text on bars
        color='is_duplicate',  # Define the column based on which to color the bars
        color_discrete_map={0: 'SkyBlue', 1: 'LightSalmon'}  # Custom colors for each value
        )
        fig.update_xaxes(
        tickmode='array',
        tickvals=[0, 1],
        ticktext=['0', '1']
        )

        # Customize layout
        fig.update_layout(
            xaxis_title="Is Duplicate", 
            yaxis_title="Count",
            plot_bgcolor='rgba(0,0,0,0)'  # Make background color transparent
        )

        # Show the plot
        st.plotly_chart(fig)




      

        # Plot the distribution of question lengths

        data = og_data.sample(n=10000, random_state=1)
        data = data.reset_index(drop=True)


        st.markdown('## Distribution of Question Lengths')
        data['Length of Question 1'] = data['question1'].apply(lambda x: len(x.split()))
        data['Length of Question 2'] = data['question2'].apply(lambda x: len(x.split()))

        # Create a single DataFrame for Plotly
        df_long = pd.melt(data, value_vars=['Length of Question 1', 'Length of Question 2'], var_name='Question', value_name='Length')

        # Plot using Plotly Express
        fig = px.histogram(df_long, x='Length', color='Question', barmode='overlay',
                       histnorm='',  # Use 'percent' for percentage, 'probability' for probability density, or '' for count
                       marginal='box',  # Or 'rug', 'violin', 'box' for the marginal plots
                       opacity=0.6,
                       labels={'Length': 'Question Length'},
                       title='Distribution of Question Lengths',
                       color_discrete_map={
                           'Length of Question 1': 'blue',  # Custom color for Question 1
                           'Length of Question 2': 'red'    # Custom color for Question 2
                       })
        fig.update_layout(xaxis_title='Question Length', yaxis_title='Frequency')
        fig.update_traces(marker_line_width=1,marker_line_color="black")

        # Show plot
        st.plotly_chart(fig)

        ############################################################################################################
        ############################################################################################################
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
        

        #plot
        st.markdown('## Most common words')
        most_common = cnt.most_common(10)
        most_common = dict(most_common)
        df_most_common = pd.DataFrame(list(most_common.items()), columns=['Word', 'Frequency'])

        # Create a bar chart with Plotly Express
        fig = px.bar(df_most_common, x='Word', y='Frequency', color='Frequency',
                    color_continuous_scale='Oranges',  # Use a color scale based on frequency
                    title='Most Common Words')

        # Customize the layout
        fig.update_layout(xaxis_title='Words', yaxis_title='Frequency',
                        coloraxis_showscale=False)  # Hide the color scale bar

        # Show the plot
        st.plotly_chart(fig)

        ############################################################################################################
        ############################################################################################################
      # Plot the distribution of common words
        def common_words(row):
            q1=set(map(lambda word: word.lower().strip(),row['question1'].split(" ")))
            q2=set(map(lambda word: word.lower().strip(),row['question2'].split(" ")))
            return len(q1 & q2)

        data['common_words'] = data.apply(common_words, axis=1)

        # Create a violin plot with Plotly Express
        fig = px.violin(data, x='is_duplicate', y='common_words', color='is_duplicate',
                        box=True,  # Display box plot inside the violin
                        points="all",  # Display all points
                        hover_data=data.columns,  # Show all data columns in hover info
                        title='Common Words in Questions')

        # Customize the layout for better readability
        fig.update_layout(
            xaxis_title='Is Duplicate',
            yaxis_title='Number of Common Words',
            legend_title='Duplicate Status'
        )

        # Customize the color scheme if desired
        fig.update_traces(side='positive', line_color='blue', line_width=2)
        fig.update_traces(side='negative', line_color='red', line_width=2)

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        ############################################################################################################
        ############################################################################################################
        #Word length differences between questions
        st.markdown('## Word Length Differences')

        # calculate word length of questions in two columns
        data['q1_wordlen'] = data['question1'].apply(lambda x: len(str(x).split()))
        data['q2_wordlen'] = data['question2'].apply(lambda x: len(str(x).split()))

        # create a new column that has difference in word length
        data['word_difference'] = abs(data['q1_wordlen'] - data['q2_wordlen'])

        # Create a violin plot with Plotly Express
        fig = px.violin(data, x='is_duplicate', y='word_difference', color='is_duplicate',
                        box=True,  # Display box plot inside the violin
                        points="all",  # Display all points
                        hover_data=data.columns,  # Show all data columns in hover info
                        title='Word Length Difference in Questions')
        
        # Customize the layout
        fig.update_layout(
            xaxis_title='Is Duplicate',
            yaxis_title='Word Length Difference',
            legend_title='Duplicate Status'
        )
        
        # Customize the color scheme if desired
        fig.update_traces(marker=dict(opacity=0.5))
        
        # Display the plot in Streamlit
        st.plotly_chart(fig)

        ############################################################################################################
        ############################################################################################################




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
            
            if probas[0][1] > 0.3:
                st.markdown('The questions are similar')
            else:
                st.markdown('The questions are not similar')
            ## Print the probability of similarity
            st.write('Probability of similarity:', probas[0][1])

            ## Print the probability of dissimilarity
            st.write('Probability of dissimilarity:', probas[0][0])

    

if __name__ == "__main__":
    main()
