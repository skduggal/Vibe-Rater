import pandas as pd

reddit_df = pd.read_csv('Reddit_Data.csv')
twitter_df = pd.read_csv('Twitter_Data.csv')

reddit_df.rename(columns={'clean_comment': 'text'}, inplace=True)
twitter_df.rename(columns={'clean_text': 'text'}, inplace=True)

merged_df = pd.concat([reddit_df, twitter_df], ignore_index=True)

def convert_category(label):
    if label == 1:
        return "positive"
    elif label == 0:
        return "neutral"
    elif label == -1:
        return "negative"
    else:
        return "unknown"

merged_df['category'] = merged_df['category'].apply(convert_category)

final_df = merged_df[['text', 'category']]

final_csv_path = '/Users/sidkduggal/Documents/Code/Vibe-Rater/merged_category_data.csv'
final_df.to_csv(final_csv_path, index=False)
print(f"Final CSV file saved to {final_csv_path}")
