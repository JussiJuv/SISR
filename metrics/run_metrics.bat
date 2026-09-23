@echo off
setlocal enabledelayedexpansion

:: Root of the metrics folder
set "REPO=C:\koulua\Dippa_koodit_test\metrics"

:: DIV2K dataset root
set "DATA_ROOT=C:\koulua\Dippa\dataset\DIV2K"

:: Ground truth HR images
set "GT_DIR=%DATA_ROOT%\DIV2K_valid_HR"

:: Model output images folder
set "BASE_DIR=C:\koulua\Dippa_koodit_test\flow_matching_NAFNet\outputs\local_test_0922ti_20.50.49,83"

:: Output directory
set "OUT_DIR=%BASE_DIR%\results"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo BASE_DIR = %BASE_DIR%
echo GT_DIR   = %GT_DIR%
echo OUT_DIR  = %OUT_DIR%

cd /d "%REPO%"
set "PYTHONPATH=%REPO%;%PYTHONPATH%"

echo Running metrics.py...

python -u metrics.py ^
  --base_dir "%BASE_DIR%" ^
  --gt_dir "%GT_DIR%" ^
  --out_dir "%OUT_DIR%"

echo.
echo Done. Results saved to %OUT_DIR%
pause
