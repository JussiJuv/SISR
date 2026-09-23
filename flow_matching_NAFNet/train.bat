@echo off
setlocal enabledelayedexpansion

:: Root of the flow_matching_NAFNet repo
set "REPO=C:\koulua\Dippa_koodit_test\flow_matching_NAFNet"

:: DIV2K dataset root
set "DATA_ROOT=C:\koulua\Dippa\dataset\DIV2K"
set "HR_TRAIN=%DATA_ROOT%\DIV2K_train_HR"
set "LR_TRAIN=%DATA_ROOT%\DIV2K_train_LR_bicubic\X4"

:: Latent AE + stats
set "LATENT_AE=%REPO%\latent_model\latent_bokeh.pth"
set "LATENT_STATS=%REPO%\latent_model\latent_stats.pt"

:: Output dir for this run (timestamped)
for /f "tokens=1-4 delims=/. " %%a in ("%date%") do set "DSTAMP=%%c%%b%%a"
set "TSTAMP=%time::=%"
set "TSTAMP=%TSTAMP: =0%"
set "OUTPUT_DIR=%REPO%\outputs\local_test_%DSTAMP%_%TSTAMP%"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Using REPO       = %REPO%
echo Using HR_TRAIN    = %HR_TRAIN%
echo Using LR_TRAIN    = %LR_TRAIN%
echo Using OUTPUT_DIR  = %OUTPUT_DIR%
echo.

python examples\image\train.py ^
  --model nafnet ^
  --dataset div2k ^
  --world_size 1 ^
  --data_path "%HR_TRAIN%" ^
  --lr_data_path "%LR_TRAIN%" ^
  --output_dir "%OUTPUT_DIR%" ^
  --batch_size 1 ^
  --decay_lr ^
  --image_size 512 ^
  --lr 2e-4 ^
  --optimizer adamw ^
  --optimizer_betas 0.9 0.99 ^
  --weight_decay 0.1 ^
  --warmup_epochs 0 ^
  --use_ema ^
  --ode_method euler ^
  --ode_options "{\"step_size\": 1}" ^
  --val_freq 0 ^
  --num_val_images 0 ^
  --eval_frequency 0 ^
  --save_frequency 100 ^
  --epochs 2 ^
  --space_mode latent ^
  --log_steps 50 ^
  --save_inference_only ^
  --latent_ae_ckpt "%LATENT_AE%" ^
  --latent_stats_path "%LATENT_STATS%" ^
  --warm_start_lr ^
  --skewed_timesteps

echo.
echo Done. Exit code: %ERRORLEVEL%
pause
