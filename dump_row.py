import pandas as pd
import json

df = pd.read_excel(r'C:\Users\User\Downloads\Research_2025\Research_2025\Research-main\backend\reviews_with_nlp_features.xlsx')
print(json.dumps(df.iloc[0].to_dict(), indent=2))
