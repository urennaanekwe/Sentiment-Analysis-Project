import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the data
df = pd.read_csv('train_data_cleaned.csv')

# Plot overall sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment', data=df, order=['Positive', 'Neutral', 'Negative'])
plt.title('Overall Sentiment Distribution')
plt.show()

# WordCloud for most common words in complaints
complaints_text = ' '.join(df['processed_complaint'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(complaints_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Consumer Complaints')
plt.show()

# Company-wise sentiment analysis
top_companies = df['Company'].value_counts().head(10).index
company_sentiments = df[df['Company'].isin(top_companies)].groupby(['Company', 'sentiment']).size().unstack()

# Plot sentiment distribution for top companies
company_sentiments.plot(kind='bar', stacked=True, figsize=(14, 7))
plt.title('Sentiment Distribution for Top 10 Companies')
plt.xlabel('Company')
plt.ylabel('Number of Complaints')
plt.show()

print("Reporting and visualization completed successfully.")
