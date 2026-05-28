# MRL Forecast Quick Start Guide

## Step 1: Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 2: Generate Dummy Data (Optional)
If you don't have real data yet:
```bash
python src/generate_dummy_data.py --output-dir data/processed --num-samples 20
```

## Step 3: Train the Model
```bash
python src/train_nowcasting_model.py \
    --data-dir data/processed \
    --epochs 5 \
    --batch-size 4 \
    --output-dir models/checkpoints
```

## Step 4: Launch Web App
```bash
export NOWCAST_MODEL_CHECKPOINT=models/checkpoints/best_model.pt
python src/web_app.py
```
Visit `http://localhost:5000` to see the UI.

## Troubleshooting
- **Error: No module named 'train_nowcasting_model'**:
  Ensure you are running the script from the project root or add `src` to your `PYTHONPATH`:
  `export PYTHONPATH=$PYTHONPATH:$(pwd)/src`
- **Memory Errors**:
  Reduce `--batch-size` or use a smaller `--hidden-channels` configuration.
