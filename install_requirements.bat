@echo off
REM Install all Python dependencies for Windows

echo ==========================================
echo Installing Python Dependencies
echo ==========================================
echo.

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip

if %ERRORLEVEL% NEQ 0 (
    echo [X] Failed to upgrade pip!
    pause
    exit /b 1
)

echo.
echo Installing requirements from requirements.txt...
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [X] Installation failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [OK] Installation Complete!
echo ==========================================
echo.
echo You can now run:
echo   - run_pyspark_pipeline.bat       (Data ingestion + feature engineering)
echo   - run_pyspark_ml_pipeline.bat    (Model training + evaluation + Tableau prep)
echo   - run_sklearn_training.bat       (Scikit-learn model training)
echo.
pause
