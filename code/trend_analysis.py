import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df = pd.read_csv('train_data_cleaned.csv')

# Convert the date column to datetime format
df['Date received'] = pd.to_datetime(df['Date received'])

# Group data by month and count the number of complaints
monthly_trends = df.groupby(df['Date received'].dt.to_period('M')).size()

# Plot the trend of customer complaints over time
plt.figure(figsize=(10, 6))
monthly_trends.plot()
plt.title('Trend of Consumer Complaints Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Complaints')
plt.grid(True)
plt.show()

# Analyze sentiment trends over time
sentiment_trends = df.groupby([df['Date received'].dt.to_period('M'), 'sentiment']).size().unstack()

# Plot the sentiment trend over time
sentiment_trends.plot(kind='bar', stacked=True, figsize=(12, 7))
plt.title('Sentiment Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Complaints')
plt.show()

print("Trend analysis completed successfully.")
