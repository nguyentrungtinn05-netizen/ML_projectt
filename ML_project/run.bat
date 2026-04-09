@echo off
REM ============================================================
REM  Student Learning Habit AI - One-Click Launcher (Windows)
REM ============================================================

title Student Learning Habit AI

echo.
echo  ==========================================
echo   AI Learning Habit Analyzer - Starting...
echo  ==========================================
echo.

REM ── Step 1: Check Python ─────────────────────────────────────
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python khong duoc tim thay!
    echo Vui long cai Python tu https://python.org
    pause
    exit /b 1
)

echo [OK] Python da duoc cai dat.

REM ── Step 2: Install / upgrade pip quietly ────────────────────
echo [INFO] Kiem tra pip...
python -m pip install --upgrade pip --quiet

REM ── Step 3: Install dependencies ─────────────────────────────
echo [INFO] Cai dat thu vien (co the mat 1-2 phut lan dau)...
pip install -r requirements.txt --quiet

IF ERRORLEVEL 1 (
    echo [ERROR] Cai dat thu vien that bai. Kiem tra ket noi mang.
    pause
    exit /b 1
)

echo [OK] Tat ca thu vien da san sang.

REM ── Step 4: Pre-train model if missing ───────────────────────
IF NOT EXIST "model.pkl" (
    echo [INFO] Chua co model.pkl - dang huan luyen mo hinh...
    python -c "
import pandas as pd, joblib
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

FEATURE_COLS = ['gender','study_hours_per_day','attendance_percentage',
    'assignment_score','midterm_score','participation_score',
    'internet_access','extra_classes','parent_education','sleep_hours']

df = pd.read_csv('Student Performance Analytics Dataset.csv')
X = df[FEATURE_COLS]; y = df['overall_score']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

base = [('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42))]
stack = StackingRegressor(estimators=base, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)
pipe = Pipeline([('scaler', StandardScaler()), ('model', stack)])
pipe.fit(X_train, y_train)
joblib.dump(pipe, 'model.pkl')
print('Model saved!')
"
    echo [OK] Mo hinh da duoc huan luyen va luu.
)

REM ── Step 5: Launch Streamlit ──────────────────────────────────
echo.
echo  ==========================================
echo   Mo trinh duyet tai: http://localhost:8501
echo   Nhan Ctrl+C de thoat
echo  ==========================================
echo.

streamlit run app.py --server.port 8501 --server.headless false

pause
