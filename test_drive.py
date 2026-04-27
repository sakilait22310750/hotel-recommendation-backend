import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

try:
    creds = service_account.Credentials.from_service_account_file(
        r"C:\Users\User\Downloads\Research-main\Research-main\backend\service-account-key.json",
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    service = build('drive', 'v3', credentials=creds)
    results = service.files().list(pageSize=1, fields="nextPageToken, files(id, name)").execute()
    print("SUCCESS: Google Drive API connected successfully. Files found:", len(results.get('files', [])))
except Exception as e:
    print("ERROR testing Google Drive API:", str(e))
