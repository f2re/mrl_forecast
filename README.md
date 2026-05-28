# MRL Forecast: Precipitation Nowcasting with ConvLSTM

This project implements a precipitation nowcasting system using Deep Learning (Convolutional LSTM). It is designed to predict radar reflectivity maps for the next hour based on past observations.

## Features

- **Deep Learning Model**: Implementation of a multi-layer ConvLSTM for spatio-temporal prediction.
- **Training Pipeline**: Script for training the model on preprocessed radar data sequences.
- **Web Interface**: Flask-based web application for visualizing predictions.
- **Data Preprocessing**: Guidelines for converting BUFR radar data to model-ready NumPy sequences.

## Project Structure

```text
mrl_forecast/
├── src/
│   ├── train_nowcasting_model.py  # Model training script
│   └── web_app.py                 # Flask web application
├── data/
│   ├── raw/                       # Original BUFR data
│   └── processed/                 # Prepared NumPy sequences (.npy)
├── models/                        # Saved model weights
├── notebooks/                     # Exploratory Data Analysis
├── static/                        # Web app assets (CSS, JS)
├── templates/                     # Flask HTML templates (optional)
├── tests/                         # Unit and integration tests
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mrl_forecast
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation

#### Synthetic Data (for testing)
```bash
python src/generate_dummy_data.py --output-dir data/processed --num-samples 50
```

#### Operational BUFR Data
If you have raw MRL BUFR files, use the integrated pipeline to decode and prepare them:
```bash
python src/make_dataset.py --bufr-dir /path/to/raw/bufr --output-dir data/processed
```
This script uses `eccodes` to decode BUFR messages and interpolates the polar radar data to a 256x256 regular grid.

### 2. Training the Model

Run the training script pointing to your processed data:

```bash
python src/train_nowcasting_model.py \
    --data-dir data/processed \
    --epochs 20 \
    --batch-size 4 \
    --output-dir models/checkpoints
```

### 3. Running the Web Application

Set the model checkpoint path and start the Flask server:

```bash
export NOWCAST_MODEL_CHECKPOINT=models/checkpoints/best_model.pt
python src/web_app.py
```

The web interface displays:
- **History**: The last 4 frames (1 hour) of observed data.
- **Forecast**: The next 4 predicted frames (1 hour) using a standard dBZ colormap.

Open `http://localhost:5000` in your browser.

## Roadmap

- [ ] Support for TrajGRU and PredRNN architectures.
- [ ] Integration of Generative Adversarial Networks (GANs) for sharper predictions.
- [ ] Direct BUFR decoding support.
- [ ] Leaflet.js integration for map overlays.

## License

MIT License
