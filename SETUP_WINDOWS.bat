@echo off
REM Windows Setup Guide - Display instructions

echo ==========================================
echo H^&M Fashion Recommendation - Windows Setup
echo ==========================================
echo.
echo This project includes both PySpark and Scikit-learn ML pipelines.
echo.
echo ==========================================
echo SETUP INSTRUCTIONS
echo ==========================================
echo.
echo 1. Install Python 3.8+ (if not already installed)
echo    Download from: https://www.python.org/downloads/
echo.
echo 2. Install Java 8 or 11 (required for PySpark)
echo    Download from: https://adoptium.net/
echo    Set JAVA_HOME environment variable
echo.
echo 3. Install dependencies:
echo    ^> install_requirements.bat
echo.
echo ==========================================
echo USAGE
echo ==========================================
echo.
echo Option 1: PySpark Pipeline (Recommended for large datasets)
echo   ^> run_pyspark_pipeline.bat         (Data ingestion + features)
echo   ^> run_pyspark_ml_pipeline.bat      (Training + evaluation + Tableau)
echo.
echo Option 2: Scikit-learn Pipeline (Faster for small datasets)
echo   ^> run_sklearn_training.bat         (All steps in one)
echo.
echo ==========================================
echo DATA REQUIREMENTS
echo ==========================================
echo.
echo Place your H^&M CSV files in: data\hnm\raw\
echo   - articles.csv
echo   - customers.csv
echo   - transactions_train.csv
echo.
echo ==========================================
echo OUTPUT LOCATIONS
echo ==========================================
echo.
echo PySpark outputs:
echo   - Models:    data\hnm\pyspark_models\
echo   - Metrics:   data\hnm\pyspark_evaluation\
echo   - Tableau:   data\hnm\pyspark_tableau\
echo.
echo Scikit-learn outputs:
echo   - Models:    data\hnm\models\
echo   - Metrics:   data\hnm\evaluation\
echo   - Tableau:   data\hnm\tableau\
echo.
echo ==========================================
echo NEXT STEPS
echo ==========================================
echo.
echo Press any key to install dependencies now, or close this window.
pause

REM Run installation
call install_requirements.bat
