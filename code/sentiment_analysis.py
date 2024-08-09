import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report

# Load the cleaned training data
train_data = pd.read_csv('train_data_cleaned.csv')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to determine sentiment based on VADER compound score
def get_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply VADER sentiment analysis to each complaint and get sentiment labels
train_data['sentiment_score'] = train_data['processed_complaint'].apply(lambda x: sid.polarity_scores(x)['compound'])
train_data['sentiment'] = train_data['sentiment_score'].apply(get_sentiment)

# Display the sentiment distribution
print(train_data['sentiment'].value_counts())

# Load the cleaned testing data
test_data = pd.read_csv('test_data_cleaned.csv')

# Apply the same sentiment analysis to the test data
test_data['sentiment_score'] = test_data['processed_complaint'].apply(lambda x: sid.polarity_scores(x)['compound'])
test_data['sentiment'] = test_data['sentiment_score'].apply(get_sentiment)

# Evaluate model performance by comparing predicted sentiments with actual labels (if available)
# Here, we assume you have an actual 'sentiment' column for testing data
# Uncomment the next line if you have true sentiment labels to compare against
# print(classification_report(test_data['true_sentiment'], test_data['sentiment']))

print("Sentiment analysis completed successfully.")
