@echo off
REM Scikit-learn Model Training and Evaluation for Windows

echo ==========================================
echo Scikit-learn ML Pipeline
echo ==========================================
echo.

echo ==========================================
echo STEP 1: Model Training
echo ==========================================
python run_training.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [X] Model training failed!
    pause
    exit /b 1
)

echo.
echo ==========================================
echo STEP 2: Model Evaluation
echo ==========================================
python evaluate_models.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [X] Model evaluation failed!
    pause
    exit /b 1
)

echo.
echo ==========================================
echo STEP 3: Tableau Data Preparation
echo ==========================================
python prepare_tableau_data.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [X] Tableau prep failed!
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [OK] SCIKIT-LEARN PIPELINE COMPLETE!
echo ==========================================
echo.
echo Output locations:
echo   - Models:        data\hnm\models\
echo   - Evaluation:    data\hnm\evaluation\
echo   - Tableau Data:  data\hnm\tableau\
echo.
pause
