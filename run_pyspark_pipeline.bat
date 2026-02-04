@echo off
REM PySpark Pipeline Runner for Windows
REM Runs data ingestion and feature engineering in sequence

echo ==========================================
echo PySpark Data Pipeline
echo ==========================================
echo.

REM Check if spark-submit is available
where spark-submit >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Using spark-submit
    set RUNNER=spark-submit
) else (
    echo [!] spark-submit not found!
    echo Running with python instead (local mode)...
    set RUNNER=python
)

echo.
echo ==========================================
echo STEP 1: Data Ingestion
echo ==========================================
%RUNNER% pyspark_data_ingestion.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [X] Data ingestion failed!
    exit /b 1
)

echo.
echo ==========================================
echo STEP 2: Feature Engineering
echo ==========================================
%RUNNER% pyspark_feature_engineering.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [X] Feature engineering failed!
    exit /b 1
)

echo.
echo ==========================================
echo [OK] PIPELINE COMPLETE!
echo ==========================================
echo.
echo Output locations:
echo   - Staging data:  data\hnm\staging\
echo   - Features:      data\hnm\pyspark_features\
echo.
pause
