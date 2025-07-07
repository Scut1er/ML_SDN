import sys
import os

# Добавляем текущую директорию в путь Python для исправления импортов
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from typing import Dict, Any
from models.types import ModelParameters, ParametersSettings, MLPredictionRequest, MLPredictionResponse
from services.training_service import TrainingService

app = Flask(__name__)
CORS(app)

# Глобальный экземпляр ML балансера
ml_balancer = None
model_path = "models/trained_model.pth"

# Инициализация ML балансера сразу
print("🚀 Запуск Flask приложения...")
print("🔄 Инициализация ML балансера при загрузке модуля...")

def initialize_ml_balancer():
    """Инициализация ML балансера с обученной моделью если доступна"""
    global ml_balancer
    
    try:
        print(f"🔄 Инициализация ML балансера...")
        print(f"🔍 Текущая рабочая директория: {os.getcwd()}")
        print(f"🔍 Проверка пути модели: {model_path}")
        print(f"🔍 Абсолютный путь модели: {os.path.abspath(model_path)}")
        print(f"🔍 Модель существует: {os.path.exists(model_path)}")
        
        # Импорт MLBalancer здесь для избежания циклических импортов
        from models.neural_network import MLBalancer as MLBalancerClass
        print("✅ Класс MLBalancer импортирован успешно")
        
        if os.path.exists(model_path):
            print(f"📦 Загрузка обученной модели из {model_path}")
            ml_balancer = MLBalancerClass(model_path)
            print(f"✅ Обученная модель загружена успешно")
        else:
            print("📦 Обученная модель не найдена, используем необученную модель")
            ml_balancer = MLBalancerClass()
            print("✅ Необученная модель инициализирована успешно")
        
        print(f"🔍 Объект ML балансера: {ml_balancer}")
        print(f"🔍 Тип ML балансера: {type(ml_balancer)}")
            
    except Exception as e:
        print(f"❌ Ошибка инициализации ML балансера: {str(e)}")
        import traceback
        print(f"❌ Трассировка: {traceback.format_exc()}")
        ml_balancer = None

# Вызов функции инициализации после определения
initialize_ml_balancer()

@app.route('/health', methods=['GET'])
def health_check():
    """Эндпоинт проверки здоровья"""
    print(f"🏥 Вызвана проверка здоровья")
    print(f"🔍 Статус ml_balancer: {ml_balancer}")
    print(f"🔍 ml_balancer не равен None: {ml_balancer is not None}")
    print(f"🔍 model_path существует: {os.path.exists(model_path)}")
    
    return jsonify({
        "status": "healthy" if ml_balancer is not None else "unhealthy",
        "service": "ML Balancer",
        "model_loaded": ml_balancer is not None,
        "trained_model_available": os.path.exists(model_path),
        "model_path": model_path,
        "current_dir": os.getcwd()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Основной эндпоинт прогнозирования для решений балансировки нагрузки"""
    try:
        print("🔍 Вызван эндпоинт predict")
        data = request.json
        print(f"📨 Получены данные: {data}")
        
        # Проверка инициализации ML балансера
        if ml_balancer is None:
            print("❌ ML балансер не инициализирован")
            return jsonify({
                "error": "ML balancer not initialized",
                "isNeedIntervene": False
            }), 500
        
        # Парсинг запроса
        print("📋 Парсинг данных запроса...")
        request_data = parse_prediction_request(data)
        print(f"✅ Запрос парсирован успешно")
        
        # Выполнение прогноза
        print("🧠 Выполнение прогноза...")
        prediction = ml_balancer.predict_with_fallback(
            request_data.parametersSettings,
            request_data.modelsParameters
        )
        print(f"✅ Прогноз выполнен: {prediction}")
        
        # Возврат ответа
        response = {
            "isNeedIntervene": prediction.isNeedIntervene,
            "sendingModelId": prediction.sendingModelId,
            "acceptingModelId": prediction.acceptingModelId,
            "confidence": prediction.confidence,
            "method": "ML" if prediction.confidence >= 0.7 else "Traditional"
        }
        print(f"📤 Возврат ответа: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Ошибка в эндпоинте predict: {str(e)}")
        print(f"❌ Тип ошибки: {type(e).__name__}")
        import traceback
        print(f"❌ Трассировка: {traceback.format_exc()}")
        
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "isNeedIntervene": False
        }), 500

@app.route('/train', methods=['POST'])
def train():
    """Обучение ML модели с предоставленными данными"""
    try:
        data = request.json
        
        # Инициализация сервиса обучения
        training_service = TrainingService(model_path)
        
        # Проверка предоставления обучающих данных
        if 'training_data' in data:
            # Использование предоставленных обучающих данных
            training_data = []
            for item in data['training_data']:
                solution = training_service._parse_network_solution(item)
                training_data.append(solution)
        else:
            # Генерация синтетических данных
            num_samples = data.get('num_samples', 1000)
            print(f"Генерация {num_samples} синтетических образцов для обучения")
            training_data = training_service.generate_synthetic_data(num_samples)
        
        # Обучение модели
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 32)
        
        history = training_service.train(
            training_data, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # Перезагрузка обученной модели
        initialize_ml_balancer()
        
        return jsonify({
            "message": "Training completed successfully",
            "history": history,
            "model_saved": True
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Training failed"
        }), 500

@app.route('/train/synthetic', methods=['POST'])
def train_synthetic():
    """Обучение только с синтетическими данными"""
    try:
        data = request.json or {}
        
        # Инициализация сервиса обучения
        training_service = TrainingService(model_path)
        
        # Генерация синтетических данных
        num_samples = data.get('num_samples', 1000)
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 32)
        
        print(f"Генерация {num_samples} синтетических образцов для обучения")
        training_data = training_service.generate_synthetic_data(num_samples)
        
        # Обучение модели
        history = training_service.train(
            training_data, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # Перезагрузка обученной модели
        initialize_ml_balancer()
        
        return jsonify({
            "message": "Training with synthetic data completed successfully",
            "samples_generated": num_samples,
            "epochs": epochs,
            "history": history,
            "model_saved": True
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Training failed"
        }), 500

@app.route('/model/status', methods=['GET'])
def model_status():
    """Получение статуса и информации о модели"""
    return jsonify({
        "model_loaded": ml_balancer is not None,
        "trained_model_available": os.path.exists(model_path),
        "model_path": model_path,
        "model_parameters": get_model_info() if ml_balancer else None
    })

def get_model_info():
    """Получение информации о модели"""
    if ml_balancer and ml_balancer.model:
        total_params = sum(p.numel() for p in ml_balancer.model.parameters())
        trainable_params = sum(p.numel() for p in ml_balancer.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_type": "LoadBalancerNN"
        }
    return None

def parse_prediction_request(data: Dict[str, Any]) -> MLPredictionRequest:
    """Парсинг запроса прогнозирования из JSON"""
    
    # Парсинг настроек параметров
    params_settings = ParametersSettings(
        criticalLoadFactor=data['parametersSettings']['criticalLoadFactor'],
        criticalPacketLost=data['parametersSettings']['criticalPacketLost'],
        criticalPing=data['parametersSettings']['criticalPing'],
        criticalJitter=data['parametersSettings']['criticalJitter']
    )
    
    # Парсинг параметров моделей
    models_params = []
    for model_data in data['modelsParameters']:
        model = ModelParameters(
            id=model_data['id'],
            loadFactor=model_data['loadFactor'],
            packetLost=model_data['packetLost'],
            ping=model_data['ping'],
            jitter=model_data['jitter'],
            CPU=model_data.get('CPU', 0.0),
            usedDiskSpace=model_data.get('usedDiskSpace', 0.0),
            memoryUsage=model_data.get('memoryUsage', 0.0),
            networkTraffic=model_data.get('networkTraffic', 0.0)
        )
        models_params.append(model)
    
    return MLPredictionRequest(
        parametersSettings=params_settings,
        modelsParameters=models_params
    )

if __name__ == '__main__':
    # Инициализация ML балансера
    initialize_ml_balancer()
    
    # Запуск Flask приложения
    app.run(host='0.0.0.0', port=5001, debug=True) 