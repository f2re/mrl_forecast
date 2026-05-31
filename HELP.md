# MRL Forecast Quick Start Guide

## Step 1: Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 2: Acquire Data
### Option A: Real Archive Data (Recommended)
```bash
# Download real NEXRAD scans
python src/download_archive.py --date 2024-05-20 --count 20 --output data/raw/archive
# Process into sequences
python src/make_dataset.py --archive-dir data/raw/archive/<session_id> --output-dir data/processed_archive --seq-len 8
```

### Option B: Generate Dummy Data (Optional)
If you just want to test the training loop quickly:
```bash
python src/generate_dummy_data.py --output-dir data/processed --num-samples 20
```

## Step 3: Train the Model
```bash
python src/train_nowcasting_model.py \
    --data-dirs data/processed_archive/<dataset_id> \
    --epochs 5 \
    --batch-size 4 \
    --output-dir models/checkpoints
```

## Step 4: Launch Web App
```bash
export NOWCAST_MODEL_CHECKPOINT=models/checkpoints/best_model.pt
python src/web_app.py
```
Visit `http://localhost:5005` to see the UI.

## Diagnostics
```bash
bash scripts/doctor.sh
python scripts/check_aws_source.py --station KOKX --date 2024-05-20
```

## Troubleshooting
- **Error: No module named 'train_nowcasting_model'**:
  Ensure you are running the script from the project root or add `src` to your `PYTHONPATH`:
  `export PYTHONPATH=$PYTHONPATH:$(pwd)/src`
- **Memory Errors**:
  Reduce `--batch-size` or use a smaller `--hidden-channels` configuration.
