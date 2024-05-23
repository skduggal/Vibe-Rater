import instaloader
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
from collections import Counter

def get_instagram_comments(post_url, username, password):
    L = instaloader.Instaloader()

    try:
        L.login(username, password)
    except instaloader.exceptions.ConnectionException as e:
        print(f"Login failed: {e}")
        return []

    shortcode = post_url.split("/")[-2]

    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
    except instaloader.exceptions.ConnectionException as e:
        print(f"Failed to fetch post: {e}")
        return []

    comments = []
    try:
        for comment in post.get_comments():
            comments.append({"author": comment.owner.username, "text": comment.text, "time": comment.created_at_utc})
            if len(comments) >= 1000:  # Stop after getting 1000 comments
                break
    except instaloader.exceptions.ConnectionException as e:
        print(f"Failed to fetch comments: {e}")
        return []

    return comments

def save_comments_to_csv(comments, filename="insta_comments.csv"):
    if not comments:
        print("No comments to save.")
        return

    df = pd.DataFrame(comments)
    df.to_csv(filename, index=False)
    print(f"Comments saved to {filename}")

def analyze_sentiment(csv_file):
    df = pd.read_csv(csv_file)
    model = BertForSequenceClassification.from_pretrained("./fine-tuned-bert")
    tokenizer = BertTokenizer.from_pretrained("./fine-tuned-bert")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    def bert_sentiment(text):
        if pd.isna(text):
            return "neutral"
        result = sentiment_pipeline(text)[0]
        if result['label'] == 'LABEL_0':  
            return 'negative'
        elif result['label'] == 'LABEL_1':
            return 'neutral'
        else:
            return 'positive'

    df['category'] = df['text'].apply(bert_sentiment)
    
    output_file = "insta_comments_with_category.csv"
    df.to_csv(output_file, index=False)
    print("Sentiment analysis complete. Results saved to insta_comments_with_category.csv")

    sentiment_counts = Counter(df['category'])
    overall_sentiment = sentiment_counts.most_common(1)[0][0]
    print(f"Overall Sentiment for the post: {overall_sentiment}")
    
    return output_file, overall_sentiment

def display_csv(csv_file):
    df = pd.read_csv(csv_file)
    print(df)

if __name__ == "__main__":
    post_url = input("Enter the Instagram post URL: ")
    username = input("Enter your Instagram username: ")
    password = input("Enter your Instagram password: ")
    comments = get_instagram_comments(post_url, username, password)
    save_comments_to_csv(comments)
    if comments:
        output_file, overall_sentiment = analyze_sentiment("insta_comments.csv")
        display_csv(output_file)
    else:
        print("No comments were fetched, skipping sentiment analysis.")
