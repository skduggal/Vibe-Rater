import instaloader
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

L = instaloader.Instaloader()

def get_instagram_comments(post_url, username, password):
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
    analyzer = SentimentIntensityAnalyzer()
    
    def vader_sentiment(text):
        if pd.isna(text):
            return 0.0
        sentiment_dict = analyzer.polarity_scores(str(text))
        return sentiment_dict['compound']
    
    df['sentiment'] = df['text'].apply(vader_sentiment)
    
    df['rating'] = df['sentiment'].apply(lambda x: int(((x + 1) / 2) * 9 + 1))
    
    output_file = "insta_comments_with_sentiment.csv"
    df.to_csv(output_file, index=False)
    print("Sentiment analysis complete. Results saved to insta_comments_with_sentiment.csv")

    overall_score = df['rating'].mean()
    print(f"Overall Score for the post: {overall_score:.2f}/10")
    
    return output_file, overall_score

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
        output_file, overall_score = analyze_sentiment("insta_comments.csv")
        display_csv(output_file)
    else:
        print("No comments were fetched, skipping sentiment analysis and download link creation.")