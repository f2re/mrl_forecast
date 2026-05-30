#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import pathlib

# Добавляем src в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Force AWS region for NEXRAD data
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

from train_nowcasting_model import ConvLSTM
from adapters import NOAAAWSAdapter, LocalDirectoryAdapter
from map_visualization import generate_sequence_plots

def main():
    parser = argparse.ArgumentParser(description="CLI для генерации MRL прогнозов (Nowcasting).")
    parser.add_argument('--model-path', type=str, required=True, help="Путь к файлу модели (например, best_model.pt)")
    parser.add_argument('--station', type=str, default='kokx', help="Код станции (например, kokx)")
    parser.add_argument('--source', type=str, choices=['ftp', 'local'], default='ftp', help="Источник данных")
    parser.add_argument('--local-dir', type=str, default='data/processed', help="Папка для source=local")
    parser.add_argument('--output-dir', type=str, default='data/predictions', help="Папка для сохранения результатов")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = args.model_path
    if os.path.isdir(model_path):
        candidate = os.path.join(model_path, 'best_model.pt')
        if os.path.exists(candidate):
            model_path = candidate
        else:
            print(f"Ошибка: В директории {model_path} не найден файл best_model.pt")
            sys.exit(1)

    print(f"Загрузка модели из {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    input_length = checkpoint.get('input_length', 4)
    target_length = checkpoint.get('target_length', 4)
    
    model = ConvLSTM(
        input_channels=1, 
        hidden_channels=checkpoint.get('hidden_channels', [32, 1]), 
        output_steps=target_length
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Загрузка последних данных ({input_length} кадров) для {args.station}...")
    if args.source == 'ftp':
        adapter = NOAAAWSAdapter()
        array, timestamps, msg = adapter.get_latest_sequence(input_length, station_code=args.station)
    else:
        adapter = LocalDirectoryAdapter(args.local_dir)
        array, timestamps, msg = adapter.get_latest_sequence(input_length)
    
    print(f"Статус данных: {msg}")

    # Предобработка
    array = np.clip(array, 0.0, 70.0) / 70.0
    tensor_input = torch.from_numpy(array).unsqueeze(1).unsqueeze(0).float().to(device)

    print("Генерация прогноза...")
    with torch.no_grad():
        preds, _ = model(tensor_input)

    in_data = tensor_input.cpu().squeeze(0).squeeze(1).numpy()
    pred_data = preds.cpu().squeeze(0).squeeze(1).numpy()

    # Сохранение сырых массивов
    np.save(os.path.join(args.output_dir, f'history_{args.station}.npy'), in_data)
    np.save(os.path.join(args.output_dir, f'forecast_{args.station}.npy'), pred_data)
    print("Тензоры сохранены в формате .npy")

    # Визуализация с картографической основой
    print("Рендеринг карт (с геопривязкой)...")
    last_ts = timestamps[-1] if timestamps else None
    png_list = generate_sequence_plots(
        in_data, pred_data, input_length, 
        station_code=args.station,
        start_datetime=last_ts,
        history_timestamps=timestamps
    )
    
    for i, png_bytes in enumerate(png_list):
        is_forecast = i >= input_length
        prefix = "forecast" if is_forecast else "history"
        idx = i - input_length if is_forecast else i
        filename = os.path.join(args.output_dir, f"{args.station}_{prefix}_{idx}.png")
        with open(filename, 'wb') as f:
            f.write(png_bytes)
            
    print(f"Готово! Результаты сохранены в {args.output_dir}")

if __name__ == '__main__':
    main()
