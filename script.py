import pandas as pd

df_reviews = pd.read_csv('Reviews.csv', encoding='utf-8')
df_reviews.sample(n=1000, random_state=42).to_csv('Reviews.csv', index=False)

df_twitter = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df_twitter.sample(n=10000, random_state=42).to_csv('training.1600000.processed.noemoticon.csv', index=False)