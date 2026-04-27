# How to Start the Backend Server

## Backend Port
The backend server runs on **port 8000** by default.

## Steps to Start Backend

### 1. Navigate to Backend Folder
```bash
cd backend
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Start the Server
```bash
python server.py
```

You should see output like:
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 4. Verify Backend is Running

Open a browser and go to:
- **API Documentation:** http://localhost:8000/docs
- **Alternative docs:** http://localhost:8000/redoc

If you see the API documentation page, the backend is running correctly!

### 5. For Physical Device Access

The backend is configured to listen on `0.0.0.0:8000`, which means it accepts connections from:
- `localhost:8000` (same computer)
- `192.168.8.178:8000` (your computer's IP on the network)

## Troubleshooting

### Port Already in Use
If you get an error that port 8000 is already in use:

1. **Find what's using the port:**
   ```powershell
   # Windows
   netstat -ano | findstr :8000
   
   # Mac/Linux
   lsof -i :8000
   ```

2. **Kill the process** (replace PID with the process ID):
   ```powershell
   # Windows
   taskkill /PID <PID> /F
   
   # Mac/Linux
   kill -9 <PID>
   ```

### Backend Won't Start
- Make sure virtual environment is activated
- Check if all dependencies are installed: `pip install -r requirements.txt`
- Check for error messages in the terminal

### Can't Access from iPhone
- Make sure both devices are on the same Wi-Fi network
- Check Windows Firewall - allow Python through firewall
- Verify backend is listening on `0.0.0.0` not `127.0.0.1` (already fixed in server.py)
