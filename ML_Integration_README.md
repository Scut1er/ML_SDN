# ML Balancer Integration Guide

## Описание

Добавлен ML-сервис для балансировки нагрузки в системе массового обслуживания. Нейросеть анализирует параметры моделей и принимает решения о перенаправлении агентов.

## Быстрый запуск

### Вариант 1: Автоматический запуск всех сервисов

```batch
run_with_ml.bat
```

### Вариант 2: Ручной запуск

1. **Установка зависимостей ML сервиса**:
   ```bash
   cd ml-balancer-service
   python run.py install
   ```

2. **Запуск ML сервиса**:
   ```bash
   python run.py run
   ```

3. **Обучение модели** (в новом терминале):
   ```bash
   python run.py train --samples 1000 --epochs 50
   ```

4. **Запуск основного проекта**:
   ```batch
   run.bat
   ```

## Как работает интеграция

### Архитектура
```
Frontend → Backend → ML Service (port 5001)
    ↓         ↓
Settings   MLBalancer
```

### Компоненты

1. **ML Service** (`ml-balancer-service/`):
   - Flask API на порту 5001
   - PyTorch нейросеть для принятия решений
   - Fallback на традиционный алгоритм

2. **MLBalancer** (`SDN-sim-mod-backend/src/domains/MLBalancer/`):
   - Интегрируется с основным бэкендом
   - Отправляет запросы в ML Service
   - Автоматически переключается на традиционный алгоритм при недоступности ML

3. **Board** (модифицирован):
   - Поддерживает переключение между обычным и ML балансировщиком
   - Настройка `isMLBalancingActive` в конфигурации

## Использование

### В настройках фронтенда:
- При включении ML режима бэкенд будет использовать MLBalancer

### Типы решений:
- **ML**: Нейросеть принимает решение с высокой уверенностью (>70%)
- **Traditional**: Fallback на традиционный алгоритм

## Данные для обучения

Пример данных для обучения:
```typescript
const trainingExample = {
  parametersSettings: {
    criticalLoadFactor: 1.0,
    criticalPacketLost: 0.1,
    criticalPing: 100,
    criticalJitter: 20
  },
  modelsParameters: [
    {
      id: "model_1",
      loadFactor: 0.75,
      packetLost: 0.05,
      ping: 50,
      jitter: 10,
      CPU: 0.6,
      usedDiskSpace: 0.4,
      memoryUsage: 0.7,
      networkTraffic: 0.8
    }
  ],
  firstWaySolution: {
    isNeedIntervene: true,
    sendingModelId: "model_1",
    acceptingModelId: "model_2"
  }
};
```

## Мониторинг

- ML Service health: `GET http://localhost:5001/health`
- Логи показывают используемый метод: `[ML]` или `[Traditional]`
- Уровень уверенности отображается в логах

## Файлы изменений

### Новые файлы:
- `ml-balancer-service/` - весь ML сервис
- `SDN-sim-mod-backend/src/domains/MLBalancer/` - ML балансировщик
- `run_with_ml.bat` - запуск с ML сервисом

### Измененные файлы:
- `SDN-sim-mod-backend/src/domains/Board/index.ts` - поддержка ML балансировщика
- `SDN-sim-mod-backend/src/domains/Board/meta.ts` - добавлен `isMLBalancingActive`