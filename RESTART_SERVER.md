# How to Restart the Backend Server

## Quick Restart (If server is already running)

1. **Stop the current server:**
   - In the terminal where the server is running, press `Ctrl+C`

2. **Restart the server:**
   - Navigate to the backend directory
   - Run the server again

## Method 1: Using Python directly (Development)

### Windows PowerShell:
```powershell
cd backend
.\venv\Scripts\Activate.ps1
python server.py
```

### Windows CMD:
```cmd
cd backend
venv\Scripts\activate
python server.py
```

### Linux/Mac:
```bash
cd backend
source venv/bin/activate
python server.py
```

## Method 2: Using uvicorn directly

### Windows PowerShell:
```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

### Windows CMD:
```cmd
cd backend
venv\Scripts\activate
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

### Linux/Mac:
```bash
cd backend
source venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

## Method 3: If using a process manager (Production)

If the server is running with supervisor or systemd:

### Supervisor:
```bash
sudo supervisorctl restart backend
```

### Systemd:
```bash
sudo systemctl restart backend
```

## Verify Server is Running

After starting, you should see output like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## Troubleshooting

### Port already in use:
If you get an error that port 8000 is already in use:
1. Find the process using port 8000:
   ```powershell
   netstat -ano | findstr :8000
   ```
2. Kill the process (replace PID with the process ID):
   ```powershell
   taskkill /PID <PID> /F
   ```
3. Restart the server

### Virtual environment not activated:
Make sure you activate the virtual environment before running the server. The `venv` folder should be in the backend directory.

### Missing dependencies:
If you get import errors, install dependencies:
```powershell
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment Variables

Make sure these are set before starting:
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to service account JSON file
- `MONGO_URL` - MongoDB connection string
- `DB_NAME` - Database name
- `SECRET_KEY` - JWT secret key

These are typically set in a `.env` file in the backend directory.






