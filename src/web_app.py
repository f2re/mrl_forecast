#!/usr/bin/env python3
"""
web_app.py
==========
Flask web application for serving precipitation nowcasting predictions.
Supports multiple data sources: local directory, online provider (RainViewer), and manual upload.
"""

import io
import os
import pathlib
import base64
import sys
from typing import List, Optional

import numpy as np
from flask import Flask, request, render_template_string, redirect, url_for

import torch
import torch.nn as nn

# Add src to path if running as a script
sys.path.append(str(pathlib.Path(__file__).parent))

from train_nowcasting_model import ConvLSTM
from adapters import LocalDirectoryAdapter, RainViewerAdapter


app = Flask(__name__)
app.config['LOCAL_DATA_DIR'] = os.environ.get('RADAR_DATA_DIR', 'data/processed')

# Configuration
CHECKPOINT_PATH = os.environ.get('NOWCAST_MODEL_CHECKPOINT', 'models/checkpoints/best_model.pt')

# Global model and settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model: Optional[nn.Module] = None
INPUT_LENGTH: int = 4
TARGET_LENGTH: int = 4
HIDDEN_CHANNELS: List[int] = []


def _load_model(checkpoint_path: str):
    """Load a ConvLSTM model checkpoint and update global settings."""
    global model, INPUT_LENGTH, TARGET_LENGTH, HIDDEN_CHANNELS
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running in UI-only mode.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    INPUT_LENGTH = checkpoint.get('input_length', 4)
    TARGET_LENGTH = checkpoint.get('target_length', 4)
    HIDDEN_CHANNELS = checkpoint.get('hidden_channels', [16, 32])
    mdl = ConvLSTM(
        input_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        kernel_size=3,
        output_steps=TARGET_LENGTH,
    )
    mdl.load_state_dict(checkpoint['model_state_dict'])
    mdl.to(device)
    mdl.eval()
    model = mdl


def _preprocess_input(array: np.ndarray) -> torch.Tensor:
    """Prepare a radar sequence for model inference."""
    assert array.ndim == 3, 'Input array must have shape (T, H, W)'
    MAX_DBZ = 70.0
    array = np.clip(array, 0.0, MAX_DBZ)
    array = array / MAX_DBZ
    # Adjust length
    if array.shape[0] > INPUT_LENGTH:
        array = array[-INPUT_LENGTH:]
    elif array.shape[0] < INPUT_LENGTH:
        pad_shape = (INPUT_LENGTH - array.shape[0],) + array.shape[1:]
        pad = np.zeros(pad_shape, dtype=array.dtype)
        array = np.concatenate([pad, array], axis=0)
    tensor = torch.from_numpy(array).unsqueeze(1).unsqueeze(0).float().to(device)
    return tensor


def _tensor_to_png_images(input_tensor: torch.Tensor, pred_tensor: torch.Tensor) -> List[bytes]:
    """Convert input and prediction tensors to a list of PNG byte strings."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    
    # Define a standard dBZ colormap
    clevs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    ccols = ['#ffffff', '#00ecec', '#01a0f6', '#0000f6', '#00ff00', '#00c800', '#009000', '#ffff00', '#e7c000', '#ff9000', '#ff0000', '#d60000', '#c00000', '#ff00ff', '#9955c9']
    cmap = colors.ListedColormap(ccols)
    norm = colors.BoundaryNorm(clevs, cmap.N)

    images: List[bytes] = []
    
    # Input sequence (Historical)
    in_data = input_tensor.cpu().squeeze(0).squeeze(1).numpy()
    for i in range(in_data.shape[0]):
        fig, ax = plt.subplots(figsize=(4, 4))
        dbz = in_data[i] * 70.0
        im = ax.imshow(dbz, cmap=cmap, norm=norm)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        buf.seek(0)
        images.append(buf.read())
        
    # Prediction sequence
    pred_data = pred_tensor.cpu().squeeze(0).squeeze(1).numpy()
    for i in range(pred_data.shape[0]):
        fig, ax = plt.subplots(figsize=(4, 4))
        dbz = pred_data[i] * 70.0
        im = ax.imshow(dbz, cmap=cmap, norm=norm)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        buf.seek(0)
        images.append(buf.read())
        
    return images


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        source_type = request.form.get('source_type')
        
        try:
            if source_type == 'upload':
                uploaded = request.files.get('file')
                if not uploaded:
                    return redirect(url_for('index'))
                data = np.load(uploaded)
                array = data['arr_0'] if isinstance(data, np.lib.npyio.NpzFile) else data
            
            elif source_type == 'local':
                path = request.form.get('local_path', app.config['LOCAL_DATA_DIR'])
                adapter = LocalDirectoryAdapter(path)
                array = adapter.get_latest_sequence(INPUT_LENGTH)
            
            elif source_type == 'online':
                adapter = RainViewerAdapter()
                array = adapter.get_latest_sequence(INPUT_LENGTH)
            
            else:
                return 'Invalid source type.', 400

            if model is None:
                return 'Model not loaded. Please ensure a valid checkpoint exists.', 500

            # Preprocess and predict
            tensor_input = _preprocess_input(array)
            with torch.no_grad():
                preds = model(tensor_input)
            
            # Get both history and prediction images
            png_list = _tensor_to_png_images(tensor_input, preds)
            
            # Build HTML to display predictions
            history_html = f'<h3>History (Source: {source_type})</h3>'
            pred_html = '<h3>Forecast (Next hour)</h3>'
            
            for idx, img_data in enumerate(png_list):
                b64 = base64.b64encode(img_data).decode('utf-8')
                if idx < INPUT_LENGTH:
                    lead_time = (INPUT_LENGTH - idx - 1) * -15
                    history_html += (
                        '<div style="display:inline-block;margin:10px;text-align:center;">'
                        f'<img src="data:image/png;base64,{b64}" alt="T{lead_time} min" width="200"/>'
                        f'<div>T{lead_time} min</div></div>'
                    )
                else:
                    lead_time = (idx - INPUT_LENGTH + 1) * 15
                    pred_html += (
                        '<div style="display:inline-block;margin:10px;text-align:center;">'
                        f'<img src="data:image/png;base64,{b64}" alt="T+{lead_time} min" width="200"/>'
                        f'<div>T+{lead_time} min</div></div>'
                    )
                    
            return render_template_string(
                '''<!doctype html>
                <title>Nowcast Predictions</title>
                <style>body { font-family: sans-serif; background: #f0f0f0; margin: 20px; }</style>
                <h1>Nowcast Predictions</h1>
                <div>{{ history|safe }}</div>
                <hr>
                <div>{{ predictions|safe }}</div>
                <p><a href="{{ url_for('index') }}">Back to Settings</a></p>
                ''',
                history=history_html,
                predictions=pred_html
            )
        except Exception as exc:
            return f'Error: {exc}', 400

    # GET request: render selection form
    return render_template_string(
        '''<!doctype html>
        <title>Radar Forecast Settings</title>
        <style>
            body { font-family: sans-serif; background: #f0f0f0; margin: 40px; }
            .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 600px; }
            .section { margin-bottom: 20px; padding-bottom: 20px; border-bottom: 1px solid #eee; }
            h2 { margin-top: 0; }
            input[type="text"] { padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
        </style>
        <div class="card">
            <h1>Radar Forecast Settings</h1>
            <form method="post" enctype="multipart/form-data">
                <div class="section">
                    <h2>1. Choose Data Source</h2>
                    <input type="radio" id="src_online" name="source_type" value="online" checked>
                    <label for="src_online"><b>Online Provider</b> (RainViewer API - Global Composite)</label><br><br>
                    
                    <input type="radio" id="src_local" name="source_type" value="local">
                    <label for="src_local"><b>Local Directory</b> (Scans for .bufr / .npy / .npz)</label><br>
                    <input type="text" name="local_path" value="{{ local_dir }}" style="width: 100%; margin-top:5px;"><br><br>
                    
                    <input type="radio" id="src_upload" name="source_type" value="upload">
                    <label for="src_upload"><b>Manual Upload</b> (.npy / .npz sequence)</label><br>
                    <input type="file" name="file" style="margin-top:5px;">
                </div>
                
                <div class="section" style="border:none;">
                    <input type="submit" value="Run Nowcast" style="padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
                </div>
            </form>
            <p><small>Model: {{ model_path }}<br>Input: {{ input_length }} frames | Output: {{ target_length }} frames</small></p>
        </div>
        ''',
        local_dir=app.config['LOCAL_DATA_DIR'],
        model_path=CHECKPOINT_PATH,
        input_length=INPUT_LENGTH,
        target_length=TARGET_LENGTH
    )


if __name__ == '__main__':
    _load_model(CHECKPOINT_PATH)
    app.run(host='0.0.0.0', port=5000)
