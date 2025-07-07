#!/usr/bin/env python3
"""
Запускальщик сервиса ML балансера
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def install_dependencies():
    """Установка требуемых зависимостей"""
    print("Установка зависимостей...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Зависимости установлены успешно!")

def train_model(num_samples=1000, epochs=50, batch_size=32):
    """Обучение ML модели с синтетическими данными"""
    print(f"Обучение модели с {num_samples} образцами, {epochs} эпохами, размер батча {batch_size}")
    
    # Использование автономного скрипта обучения
    import subprocess
    
    try:
        result = subprocess.run([
            sys.executable, "train_standalone.py",
            "--samples", str(num_samples),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size)
        ], check=True, capture_output=True, text=True)
        
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Обучение не удалось: {e.stderr}")
        print(f"Вывод обучения: {e.stdout}")
        return False

def run_service():
    """Запуск сервиса ML балансера"""
    print("Запуск сервиса ML балансера...")
    
    # Добавление текущей директории в путь Python
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Установка переменных окружения
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    # Запуск Flask приложения
    from app import app
    app.run(host='0.0.0.0', port=5001, debug=True)

def main():
    parser = argparse.ArgumentParser(description="ML Balancer Service")
    parser.add_argument("command", choices=["install", "train", "run"], 
                       help="Команда для выполнения")
    parser.add_argument("--samples", type=int, default=1000, 
                       help="Количество обучающих образцов (для команды train)")
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Количество эпох обучения (для команды train)")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Размер батча для обучения (для команды train)")
    
    args = parser.parse_args()
    
    if args.command == "install":
        install_dependencies()
    elif args.command == "train":
        train_model(args.samples, args.epochs, args.batch_size)
    elif args.command == "run":
        run_service()

if __name__ == "__main__":
    main() 