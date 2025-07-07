# ML Balancer Service

Сервис машинного обучения для балансировки нагрузки в системе массового обслуживания.

## Быстрый старт

### 1. Установка зависимостей

```bash
cd ml-balancer-service
python run.py install
```

### 2. Запуск сервиса

```bash
python run.py run
```

Сервис будет доступен на `http://localhost:5001`

### 3. Обучение модели

```bash
# В новом терминале (пока сервис запущен)
python run.py train --samples 1000 --epochs 50 --batch-size 32
```

## API Endpoints

### Health Check
```
GET /health
```

### Предсказание балансировки
```
POST /predict
Content-Type: application/json

{
  "parametersSettings": {
    "criticalLoadFactor": 1.0,
    "criticalPacketLost": 0.1,
    "criticalPing": 100,
    "criticalJitter": 20
  },
  "modelsParameters": [
    {
      "id": "model_1",
      "loadFactor": 0.75,
      "packetLost": 0.05,
      "ping": 50,
      "jitter": 10,
      "CPU": 0.6,
      "usedDiskSpace": 0.4,
      "memoryUsage": 0.7,
      "networkTraffic": 0.8
    },
    {
      "id": "model_2", 
      "loadFactor": 0.45,
      "packetLost": 0.02,
      "ping": 30,
      "jitter": 5,
      "CPU": 0.3,
      "usedDiskSpace": 0.2,
      "memoryUsage": 0.4,
      "networkTraffic": 0.5
    }
  ]
}
```

Response:
```json
{
  "isNeedIntervene": true,
  "sendingModelId": "model_1",
  "acceptingModelId": "model_2",
  "confidence": 0.85,
  "method": "ML"
}
```

### Обучение модели
```
POST /train/synthetic
Content-Type: application/json

{
  "num_samples": 1000,
  "epochs": 50,
  "batch_size": 32
}
```

### Статус модели
```
GET /model/status
```

## Архитектура

### Нейронная сеть
- **Входные данные**: 36 признаков (параметры критических значений + параметры моделей)
- **Архитектура**: 3 слоя с ReLU активацией и dropout
- **Выходы**: 
  - Решение о балансировке (binary classification)
  - Выбор отправляющей модели (softmax)
  - Выбор принимающей модели (softmax)

### Fallback механизм
Если уверенность ML модели < 70%, сервис автоматически переключается на традиционный алгоритм балансировки.

## Интеграция с основным проектом

1. В настройках фронтенда добавить переключатель ML балансировки
2. Бэкенд использует `MLBalancer` класс вместо `Balancer` при активации ML режима
3. ML сервис должен быть запущен перед включением ML режима

## Обучение

Сервис поддерживает:
- Обучение на синтетических данных
- Обучение на реальных данных (передача через API)
- Автоматическое сохранение/загрузка обученной модели

## Файлы

- `app.py` - Flask API сервер
- `models/neural_network.py` - Нейронная сеть и ML балансировщик
- `models/types.py` - Типы данных
- `services/training_service.py` - Сервис обучения
- `run.py` - Скрипт запуска 