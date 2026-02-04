@echo off
REM PySpark ML Pipeline Runner for Windows
REM Trains models, evaluates them, and prepares Tableau data

echo ==========================================
echo PySpark ML Pipeline
echo ==========================================
echo.

REM Check if PySpark is available
where spark-submit >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Using spark-submit
    set RUNNER=spark-submit
) else (
    echo Running with python (local mode)...
    set RUNNER=python
)

echo.
echo ==========================================
echo STEP 1: Model Training
echo ==========================================
%RUNNER% pyspark_model_training.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [X] Model training failed!
    exit /b 1
)

echo.
echo ==========================================
echo STEP 2: Model Evaluation
echo ==========================================
%RUNNER% pyspark_model_evaluation.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [X] Model evaluation failed!
    exit /b 1
)

echo.
echo ==========================================
echo STEP 3: Tableau Data Preparation
echo ==========================================
%RUNNER% pyspark_tableau_prep.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [X] Tableau prep failed!
    exit /b 1
)

echo.
echo ==========================================
echo [OK] ML PIPELINE COMPLETE!
echo ==========================================
echo.
echo Output locations:
echo   - Models:        data\hnm\pyspark_models\
echo   - Evaluation:    data\hnm\pyspark_evaluation\
echo   - Tableau Data:  data\hnm\pyspark_tableau\
echo.
pause
