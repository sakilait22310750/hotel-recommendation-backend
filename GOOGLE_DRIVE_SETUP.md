# Google Drive API Setup Guide

This guide explains how to set up Google Drive API integration for serving hotel images.

## Prerequisites

1. Google Cloud Project with Google Drive API enabled
2. Service Account credentials (recommended) or OAuth2 credentials

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Google Drive API**:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click "Enable"

## Step 2: Create Service Account (Recommended)

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "Service Account"
3. Fill in the service account details:
   - Name: `hotel-images-service`
   - Description: `Service account for accessing hotel images`
4. Click "Create and Continue"
5. Skip role assignment (or assign "Editor" if needed)
6. Click "Done"

## Step 3: Create Service Account Key

1. Click on the created service account
2. Go to "Keys" tab
3. Click "Add Key" > "Create new key"
4. Choose "JSON" format
5. Download the JSON file
6. **Important**: Keep this file secure and never commit it to version control

## Step 4: Share Google Drive Folder with Service Account

1. Open the Google Drive folder: https://drive.google.com/drive/folders/1LwQm93QxqnwWTGv75xCejyu8a3iT7SGn
2. Click "Share" button
3. Get the service account email from the JSON file (field: `client_email`)
4. Add the service account email with "Viewer" permissions
5. Click "Send"

## Step 5: Configure Environment Variables

Add the following to your `.env` file in the backend directory:

```env
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

Or set it as an environment variable:

**Windows (PowerShell):**
```powershell
$env:GOGLE_APPLICATION_CREDENTIALS=O"C:\path\to\service-account-key.json"
```

**Windows (CMD):**
```cmd
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account-key.json
```

**Linux/Mac:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

## Step 6: Install Dependencies

The required packages are already in `requirements.txt`. Install them:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib
```

## Step 7: Test the Integration

1. Start your backend server
2. Test the endpoint:
   ```bash
   curl http://localhost:8000/api/hotel-images/1
   ```
   This should redirect to the first image for hotel ID 1

## Troubleshooting

### Error: "Google Drive service not available"
- Check that `GOOGLE_APPLICATION_CREDENTIALS` is set correctly
- Verify the JSON file path is correct
- Ensure the JSON file is valid

### Error: "Hotel folder not found"
- Verify the hotel folder exists in Google Drive
- Check that the folder name matches the hotel_id exactly
- Ensure the service account has access to the folder

### Error: "No images found"
- Check that images exist in the hotel folder
- Verify image file names (should be 1.jpg, 2.jpg, etc.)
- Ensure images are not in trash

### Error: "Permission denied"
- Verify the service account email has "Viewer" access to the folder
- Check that the folder is shared with the service account

## Alternative: OAuth2 Setup (Not Recommended for Server)

If you need to use OAuth2 instead of service account:

1. Create OAuth2 credentials in Google Cloud Console
2. Download the client secret JSON
3. Implement OAuth2 flow in the backend
4. Store and refresh tokens appropriately

Note: Service Account is recommended for server-to-server authentication.

## Security Notes

- **Never commit** the service account JSON file to version control
- Add `*.json` (service account files) to `.gitignore`
- Use environment variables for sensitive paths
- Rotate service account keys periodically
- Limit service account permissions to minimum required (Viewer only)

