import pandas as pd

df = pd.read_excel(r'C:\Users\User\Downloads\Research_2025\Research_2025\Research-main\backend\reviews_with_nlp_features.xlsx')
uga_df = df[df['hotel_name'].str.contains('Uga Chena', case=False, na=False)]

print("Excel reviews for Uga Chena Huts:")
for _, row in uga_df.head(10).iterrows():
    print(f"uid: {row.get('user_id')} | lang: {row.get('language')} | text: {str(row.get('review', ''))[:40]}")
