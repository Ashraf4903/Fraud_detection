@echo off
title Fraud Detection - Setup
color 0A

echo.
echo  ====================================================
echo    FRAUD DETECTION SYSTEM - WINDOWS SETUP
echo  ====================================================
echo.

:: ============================================================
:: STEP 1 - Check Python
:: ============================================================
echo  [STEP 1] Checking Python...
echo  -----------------------------------------------
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Python not found!
    echo  Download from: https://www.python.org/downloads/
    echo  Make sure to tick "Add Python to PATH" during install
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo  Found: %%i
echo.

:: ============================================================
:: STEP 2 - Create Folder Structure
:: ============================================================
echo  [STEP 2] Creating folder structure...
echo  -----------------------------------------------
if not exist "data"         mkdir data
if not exist "model"        mkdir model
if not exist "model\plots"  mkdir model\plots
if not exist "api"          mkdir api
if not exist "frontend"     mkdir frontend
if not exist ".vscode"      mkdir .vscode
echo  Folders created!
echo.

:: ============================================================
:: STEP 3 - Create Virtual Environment
:: ============================================================
echo  [STEP 3] Creating virtual environment...
echo  -----------------------------------------------
if exist "venv" (
    echo  venv already exists, skipping...
) else (
    python -m venv venv
    echo  venv created!
)
echo.

:: ============================================================
:: STEP 4 - Activate venv
:: ============================================================
echo  [STEP 4] Activating venv...
echo  -----------------------------------------------
call venv\Scripts\activate.bat
echo  venv activated!
echo.

:: ============================================================
:: STEP 5 - Upgrade pip
:: ============================================================
echo  [STEP 5] Upgrading pip...
echo  -----------------------------------------------
python -m pip install --upgrade pip --quiet
echo  pip upgraded!
echo.

:: ============================================================
:: STEP 6 - Install All Libraries
:: ============================================================
echo  [STEP 6] Installing all libraries (takes 2-3 mins)...
echo  -----------------------------------------------
echo  Installing numpy...
pip install numpy==1.26.4 --quiet
echo  Installing pandas...
pip install pandas==2.2.1 --quiet
echo  Installing scikit-learn...
pip install scikit-learn==1.4.1 --quiet
echo  Installing imbalanced-learn...
pip install imbalanced-learn==0.12.0 --quiet
echo  Installing joblib...
pip install joblib==1.3.2 --quiet
echo  Installing matplotlib...
pip install matplotlib==3.8.3 --quiet
echo  Installing seaborn...
pip install seaborn==0.13.2 --quiet
echo  Installing plotly...
pip install plotly==5.20.0 --quiet
echo  Installing fastapi...
pip install fastapi==0.110.0 --quiet
echo  Installing uvicorn...
pip install "uvicorn[standard]==0.28.0" --quiet
echo  Installing pydantic...
pip install pydantic==2.6.3 --quiet
echo  Installing streamlit...
pip install streamlit==1.32.2 --quiet
echo  Installing requests...
pip install requests==2.31.0 --quiet
echo  Installing python-dotenv...
pip install python-dotenv==1.0.1 --quiet
echo  Installing python-json-logger...
pip install python-json-logger==2.0.7 --quiet
echo.
echo  All libraries installed!
echo.

:: ============================================================
:: STEP 7 - Create .env file
:: ============================================================
echo  [STEP 7] Creating .env file...
echo  -----------------------------------------------
if not exist ".env" (
    echo FRAUD_API_KEY=demo-api-key-change-in-production> .env
    echo API_URL=http://localhost:8000>> .env
    echo APP_ENV=development>> .env
    echo  .env created!
) else (
    echo  .env already exists, skipping...
)
echo.

:: ============================================================
:: STEP 8 - Create VS Code settings
:: ============================================================
echo  [STEP 8] Creating VS Code settings...
echo  -----------------------------------------------
echo { > .vscode\settings.json
echo     "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe", >> .vscode\settings.json
echo     "python.terminal.activateEnvironment": true >> .vscode\settings.json
echo } >> .vscode\settings.json
echo  VS Code settings created!
echo.

:: ============================================================
:: STEP 9 - Generate Dataset
:: ============================================================
echo  [STEP 9] Generating synthetic dataset...
echo  -----------------------------------------------
if exist "data\creditcard.csv" (
    echo  Dataset already exists, skipping...
) else (
    python -c "
import numpy as np, pandas as pd
np.random.seed(42)
N_LEGIT, N_FRAUD = 49150, 850
def make_legit(n):
    X = np.random.randn(n, 28) * 0.8
    X[:, 0] = np.random.normal(-0.5, 1.5, n)
    X[:, 3] = np.random.normal(0.8, 1.0, n)
    return X, np.random.uniform(0, 172800, n), np.abs(np.random.lognormal(3.5, 1.2, n))
def make_fraud(n):
    X = np.random.randn(n, 28) * 1.5
    X[:, 0]  = np.random.normal(-3.5, 2.0, n)
    X[:, 3]  = np.random.normal(4.5, 1.5, n)
    X[:, 10] = np.random.normal(-5.0, 2.0, n)
    X[:, 13] = np.random.normal(-6.0, 2.0, n)
    a = np.where(np.random.rand(n) < 0.5, np.random.uniform(0.5, 5.0, n), np.random.uniform(500, 2500, n))
    return X, np.random.uniform(0, 172800, n), a
lX,lt,la = make_legit(N_LEGIT)
fX,ft,fa = make_fraud(N_FRAUD)
X_all = np.vstack([lX,fX])
t_all = np.concatenate([lt,ft])
a_all = np.concatenate([la,fa])
y_all = np.concatenate([np.zeros(N_LEGIT), np.ones(N_FRAUD)])
idx = np.random.permutation(len(y_all))
X_all,t_all,a_all,y_all = X_all[idx],t_all[idx],a_all[idx],y_all[idx]
v_cols = {f'V{i+1}': X_all[:,i] for i in range(28)}
df = pd.DataFrame({'Time':t_all,**v_cols,'Amount':a_all,'Class':y_all.astype(int)})
df.to_csv('data/creditcard.csv', index=False)
print(f'Dataset saved: {len(df)} rows, fraud: {y_all.mean()*100:.2f}%%')
"
)
echo.

:: ============================================================
:: STEP 10 - Train the Model
:: ============================================================
echo  [STEP 10] Training ML models (takes 1-2 mins)...
echo  -----------------------------------------------
python model\train.py
echo.

:: ============================================================
:: DONE
:: ============================================================
echo.
echo  ====================================================
echo    SETUP COMPLETE! HOW TO RUN:
echo  ====================================================
echo.
echo  Open 2 terminals in VS Code and run:
echo.
echo  Terminal 1 (API):
echo  -----------------
echo  venv\Scripts\activate
echo  uvicorn api.main:app --reload --port 8000
echo.
echo  Terminal 2 (Frontend):
echo  ----------------------
echo  venv\Scripts\activate
echo  streamlit run frontend/app.py
echo.
echo  Then open browser:
echo  Streamlit  --^>  http://localhost:8501
echo  API Docs   --^>  http://localhost:8000/docs
echo.
pause
