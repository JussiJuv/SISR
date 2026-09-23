@echo off
setlocal enabledelayedexpansion

:: Root of the flow_matching_NAFNet repo
set "REPO=C:\koulua\Dippa_koodit_test\flow_matching_NAFNet"

:: DIV2K dataset root
set "DATA_ROOT=C:\koulua\Dippa\dataset\DIV2K"
set "HR_VAL=%DATA_ROOT%\DIV2K_valid_HR"
set "LR_VAL=%DATA_ROOT%\DIV2K_valid_LR_bicubic\X4"

:: Latent AE + stats
set "LATENT_AE=%REPO%\latent_model\latent_bokeh.pth"
set "LATENT_STATS=%REPO%\latent_model\latent_stats.pt"

set "NUM_VAL_IMAGES=5"

cd /d "%REPO%"
set "PYTHONPATH=%REPO%;%PYTHONPATH%"

:: EDIT THIS to point at the output folder from the train run
:: you want to validate (e.g. one printed by train_local_windows.bat)
set "CKPT_DIR=%REPO%\outputs\local_test_0922ti_20.50.49,83"
echo Running validation over specific checkpoints in %CKPT_DIR%

set "TARGET_CHECKPOINTS=checkpoint-1.pth"

for %%C in (%TARGET_CHECKPOINTS%) do (
  set "CKPT_NAME=%%C"
  set "CKPT_PATH=%CKPT_DIR%\%%C"

  if not exist "!CKPT_PATH!" (
    echo WARNING: !CKPT_PATH! not found! Skipping...
  ) else (
    for %%B in ("%%C") do set "CKPT_BASE=%%~nB"
    set "OUTPUT_DIR=%CKPT_DIR%\!CKPT_BASE!_val"
    if not exist "!OUTPUT_DIR!" mkdir "!OUTPUT_DIR!"

    echo ----------------------------------------
    echo Checkpoint: %%C
    echo Saving to: !OUTPUT_DIR!

    python validation.py ^
      --checkpoint "!CKPT_PATH!" ^
      --output_dir "!OUTPUT_DIR!" ^
      --dataset div2k ^
      --model nafnet ^
      --data_path "%HR_VAL%" ^
      --lr_data_path "%LR_VAL%" ^
      --num_val_images %NUM_VAL_IMAGES% ^
      --device cuda ^
      --use_ema ^
      --ode_method euler ^
      --ode_options "{\"step_size\": 0.2}" ^
      --cfg_scale 0 ^
      --warm_start_lr ^
      --space_mode latent ^
      --latent_ae_ckpt "%LATENT_AE%" ^
      --latent_stats_path "%LATENT_STATS%"
  )
)

echo.
echo Validation run complete. Results in %CKPT_DIR%
pause
