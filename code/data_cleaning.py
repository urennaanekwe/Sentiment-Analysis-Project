import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load the dataset into a pandas DataFrame
# Ensure the file path is correct, and the data is in CSV format
df = pd.read_csv('consumer_complaints.csv')

# Display the first few rows of the DataFrame to understand the structure
print(df.head())

# Handle missing data by dropping rows with missing values
# This can be modified to fill missing data if necessary
df = df.dropna()

# Remove duplicates from the DataFrame
df = df.drop_duplicates()

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs from the text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation and special characters
    text = re.sub(r'\W', ' ', text)
    # Remove numbers from the text
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply text preprocessing to the 'Consumer Complaint' column
df['processed_complaint'] = df['Consumer Complaint'].apply(preprocess_text)

# Split the data into training and testing sets for later use in model training
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Save the cleaned and processed data to new CSV files
train_data.to_csv('train_data_cleaned.csv', index=False)
test_data.to_csv('test_data_cleaned.csv', index=False)

print("Data cleaning and preprocessing completed successfully.")
