#!/usr/bin/env python3
"""
Тестовые данные для сервиса ML балансера
"""

import requests
import json
import time

# Примеры тестовых данных
test_examples = [
    {
        "name": "Высокая нагрузка модель 1",
        "data": {
            "parametersSettings": {
                "criticalLoadFactor": 1.0,
                "criticalPacketLost": 0.1,
                "criticalPing": 100,
                "criticalJitter": 20
            },
            "modelsParameters": [
                {
                    "id": "model_1",
                    "loadFactor": 1.2,  # Выше критического
                    "packetLost": 0.05,
                    "ping": 50,
                    "jitter": 10,
                    "CPU": 0.8,
                    "usedDiskSpace": 0.6,
                    "memoryUsage": 0.9,
                    "networkTraffic": 0.85
                },
                {
                    "id": "model_2",
                    "loadFactor": 0.4,  # Ниже критического
                    "packetLost": 0.02,
                    "ping": 30,
                    "jitter": 5,
                    "CPU": 0.3,
                    "usedDiskSpace": 0.2,
                    "memoryUsage": 0.4,
                    "networkTraffic": 0.35
                }
            ]
        }
    },
    {
        "name": "Проблемы QoS модель 3",
        "data": {
            "parametersSettings": {
                "criticalLoadFactor": 1.0,
                "criticalPacketLost": 0.1,
                "criticalPing": 100,
                "criticalJitter": 20
            },
            "modelsParameters": [
                {
                    "id": "model_1",
                    "loadFactor": 0.8,
                    "packetLost": 0.05,
                    "ping": 45,
                    "jitter": 8,
                    "CPU": 0.6,
                    "usedDiskSpace": 0.4,
                    "memoryUsage": 0.7,
                    "networkTraffic": 0.6
                },
                {
                    "id": "model_2",
                    "loadFactor": 0.6,
                    "packetLost": 0.03,
                    "ping": 35,
                    "jitter": 6,
                    "CPU": 0.4,
                    "usedDiskSpace": 0.3,
                    "memoryUsage": 0.5,
                    "networkTraffic": 0.45
                },
                {
                    "id": "model_3",
                    "loadFactor": 0.9,
                    "packetLost": 0.15,  # Выше критического
                    "ping": 150,         # Выше критического
                    "jitter": 25,        # Выше критического
                    "CPU": 0.85,
                    "usedDiskSpace": 0.7,
                    "memoryUsage": 0.8,
                    "networkTraffic": 0.9
                }
            ]
        }
    },
    {
        "name": "Сбалансированная система",
        "data": {
            "parametersSettings": {
                "criticalLoadFactor": 1.0,
                "criticalPacketLost": 0.1,
                "criticalPing": 100,
                "criticalJitter": 20
            },
            "modelsParameters": [
                {
                    "id": "model_1",
                    "loadFactor": 0.7,
                    "packetLost": 0.04,
                    "ping": 40,
                    "jitter": 8,
                    "CPU": 0.5,
                    "usedDiskSpace": 0.3,
                    "memoryUsage": 0.6,
                    "networkTraffic": 0.55
                },
                {
                    "id": "model_2",
                    "loadFactor": 0.6,
                    "packetLost": 0.03,
                    "ping": 35,
                    "jitter": 6,
                    "CPU": 0.4,
                    "usedDiskSpace": 0.25,
                    "memoryUsage": 0.5,
                    "networkTraffic": 0.45
                },
                {
                    "id": "model_3",
                    "loadFactor": 0.65,
                    "packetLost": 0.035,
                    "ping": 38,
                    "jitter": 7,
                    "CPU": 0.45,
                    "usedDiskSpace": 0.28,
                    "memoryUsage": 0.55,
                    "networkTraffic": 0.5
                }
            ]
        }
    }
]

def test_ml_service():
    """Тестирование ML сервиса с примерными данными"""
    base_url = "http://localhost:5001"
    
    # Проверка здоровья
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ ML сервис работает")
            print(f"Статус: {response.json()}")
        else:
            print("❌ Проверка здоровья ML сервиса не удалась")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Невозможно подключиться к ML сервису: {e}")
        return False
    
    print("\n" + "="*50)
    print("Тестирование ML прогнозов")
    print("="*50)
    
    for example in test_examples:
        print(f"\n🧪 Тестирование: {example['name']}")
        print("-" * 30)
        
        try:
            response = requests.post(
                f"{base_url}/predict",
                json=example['data'],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Решение: {'✅ Вмешаться' if result['isNeedIntervene'] else '❌ Не вмешиваться'}")
                if result['isNeedIntervene']:
                    print(f"От: {result['sendingModelId']} → К: {result['acceptingModelId']}")
                    print(f"Уверенность: {result['confidence']:.2%}")
                    print(f"Метод: {result['method']}")
                else:
                    print(f"Уверенность: {result['confidence']:.2%}")
                    print(f"Метод: {result['method']}")
            else:
                print(f"❌ Запрос не удался: {response.status_code}")
                print(response.text)
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка запроса: {e}")
        
        time.sleep(1)  # Небольшая задержка между запросами
    
    return True

def generate_training_data():
    """Генерация синтетических обучающих данных"""
    base_url = "http://localhost:5001"
    
    print("\n" + "="*50)
    print("Генерация обучающих данных")
    print("="*50)
    
    training_config = {
        "num_samples": 500,
        "epochs": 30,
        "batch_size": 16
    }
    
    try:
        response = requests.post(
            f"{base_url}/train/synthetic",
            json=training_config,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Обучение завершено успешно!")
            print(f"Образцы: {result['samples_generated']}")
            print(f"Эпохи: {result['epochs']}")
            print(f"Модель сохранена: {result['model_saved']}")
        else:
            print(f"❌ Обучение не удалось: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка обучения: {e}")

def main():
    """Основная функция тестирования"""
    print("🚀 Тестирование сервиса ML балансера")
    print("=" * 50)
    
    # Тестирование прогнозов
    if test_ml_service():
        print("\n✅ Все тесты пройдены!")
        
        # Опционально генерируем обучающие данные
        user_input = input("\nХотите сгенерировать обучающие данные? (y/n): ")
        if user_input.lower() == 'y':
            generate_training_data()
    else:
        print("\n❌ Тесты не удались!")
        print("Убедитесь, что ML сервис запущен: python run.py run")

if __name__ == "__main__":
    main() 