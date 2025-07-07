import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from .types import ModelParameters, ParametersSettings, MLPredictionResponse

class LoadBalancerNN(nn.Module):
    def __init__(self, input_size: int = 36, hidden_size: int = 128):
        super(LoadBalancerNN, self).__init__()
        
        # Экстрактор признаков
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU()
        )
        
        # Голова решения - бинарная классификация (нужна балансировка или нет)
        self.decision_head = nn.Linear(hidden_size // 4, 1)
        
        # Головы выбора модели - для выбора отправляющей и принимающей моделей
        self.sending_model_head = nn.Linear(hidden_size // 4, 10)  # Максимум 10 моделей
        self.receiving_model_head = nn.Linear(hidden_size // 4, 10)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Решение: нужна балансировка или нет
        decision = torch.sigmoid(self.decision_head(features))
        
        # Выбор модели (softmax для распределения вероятности)
        sending_probs = F.softmax(self.sending_model_head(features), dim=1)
        receiving_probs = F.softmax(self.receiving_model_head(features), dim=1)
        
        return decision, sending_probs, receiving_probs

class MLBalancer:
    def __init__(self, model_path: Optional[str] = None):
        self.model = LoadBalancerNN()
        self.model.eval()
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    def _prepare_features(self, parameters_settings: ParametersSettings, 
                         models_parameters: List[ModelParameters]) -> torch.Tensor:
        """Подготовка вектора признаков из входных данных"""
        features = []
        
        # Критические пороги (4 признака)
        features.extend([
            parameters_settings.criticalLoadFactor,
            parameters_settings.criticalPacketLost,
            parameters_settings.criticalPing,
            parameters_settings.criticalJitter
        ])
        
        # Параметры моделей (дополняем до максимум 4 моделей, 8 признаков каждая = 32 признака)
        max_models = 4
        for i in range(max_models):
            if i < len(models_parameters):
                model = models_parameters[i]
                features.extend([
                    model.loadFactor,
                    model.packetLost,
                    model.ping,
                    model.jitter,
                    model.CPU,
                    model.usedDiskSpace,
                    model.memoryUsage,
                    model.networkTraffic
                ])
            else:
                # Дополняем нулями если моделей меньше
                features.extend([0.0] * 8)
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def predict(self, parameters_settings: ParametersSettings,
                models_parameters: List[ModelParameters]) -> MLPredictionResponse:
        """Прогнозирование решения о балансировке нагрузки"""
        
        # Подготовка входных признаков
        features = self._prepare_features(parameters_settings, models_parameters)
        
        # Получение прогнозов модели
        with torch.no_grad():
            decision, sending_probs, receiving_probs = self.model(features)
            
            # Извлечение прогнозов
            need_balancing = decision.item() > 0.5
            confidence = decision.item()
            
            if need_balancing and len(models_parameters) > 1:
                # Получение индексов моделей из нейронной сети
                nn_sending_idx = torch.argmax(sending_probs[0][:len(models_parameters)]).item()
                nn_receiving_idx = torch.argmax(receiving_probs[0][:len(models_parameters)]).item()
                
                # Применение логической валидации - убеждаемся, что решение имеет смысл
                sending_candidate = models_parameters[nn_sending_idx]
                receiving_candidate = models_parameters[nn_receiving_idx]
                
                # Если НС предлагает отправку от модели с низкой нагрузкой к высоконагруженной, пытаемся исправить
                if (sending_candidate.loadFactor < receiving_candidate.loadFactor and 
                    abs(sending_candidate.loadFactor - receiving_candidate.loadFactor) > 0.3):
                    
                    # Найти наиболее проблемную модель (высокая нагрузка ИЛИ плохой QoS)
                    def problem_score(model):
                        score = 0
                        # Вес коэффициента нагрузки
                        if model.loadFactor > parameters_settings.criticalLoadFactor:
                            score += (model.loadFactor - parameters_settings.criticalLoadFactor) * 10
                        else:
                            score += model.loadFactor * 5
                        
                        # Веса QoS
                        if model.packetLost > parameters_settings.criticalPacketLost:
                            score += (model.packetLost - parameters_settings.criticalPacketLost) * 5
                        if model.ping > parameters_settings.criticalPing:
                            score += (model.ping - parameters_settings.criticalPing) / parameters_settings.criticalPing * 5
                        if model.jitter > parameters_settings.criticalJitter:
                            score += (model.jitter - parameters_settings.criticalJitter) / parameters_settings.criticalJitter * 5
                        
                        return score
                    
                    # Найти лучшую принимающую модель (низкая нагрузка + хороший QoS)
                    def receiving_score(model):
                        if model.id == models_parameters[nn_sending_idx].id:
                            return float('inf')
                        
                        score = model.loadFactor  # Предпочитаем низкую нагрузку
                        
                        # Штраф за плохой QoS
                        if model.packetLost > parameters_settings.criticalPacketLost * 0.7:
                            score += 2
                        if model.ping > parameters_settings.criticalPing * 0.7:
                            score += 2
                        if model.jitter > parameters_settings.criticalJitter * 0.7:
                            score += 2
                        
                        return score
                    
                    # Пересчет индексов на основе логических критериев
                    sending_idx = max(range(len(models_parameters)), 
                                    key=lambda i: problem_score(models_parameters[i]))
                    receiving_idx = min(range(len(models_parameters)), 
                                      key=lambda i: receiving_score(models_parameters[i]))
                    
                    # Корректировка уверенности в зависимости от того, насколько нам пришлось исправлять
                    if sending_idx != nn_sending_idx or receiving_idx != nn_receiving_idx:
                        confidence *= 0.7  # Снижаем уверенность при переопределении НС
                else:
                    sending_idx = nn_sending_idx
                    receiving_idx = nn_receiving_idx
                
                # Финальная валидация - убеждаемся, что не отправляем в ту же модель
                if sending_idx == receiving_idx:
                    receiving_probs_copy = receiving_probs[0].clone()
                    receiving_probs_copy[sending_idx] = 0
                    receiving_idx = torch.argmax(receiving_probs_copy[:len(models_parameters)]).item()
                
                return MLPredictionResponse(
                    isNeedIntervene=True,
                    sendingModelId=models_parameters[sending_idx].id,
                    acceptingModelId=models_parameters[receiving_idx].id,
                    confidence=confidence
                )
            else:
                return MLPredictionResponse(
                    isNeedIntervene=False,
                    confidence=confidence
                )
    
    def _calculate_load_score(self, model: ModelParameters, 
                            settings: ParametersSettings) -> float:
        """Вычисление балла нагрузки с использованием традиционного подхода для сравнения"""
        score = 0.0
        
        # Компонент коэффициента нагрузки
        if model.loadFactor > settings.criticalLoadFactor:
            score += (model.loadFactor - settings.criticalLoadFactor) * 10
        
        # Компоненты QoS
        if model.packetLost > settings.criticalPacketLost:
            score += (model.packetLost - settings.criticalPacketLost) * 5
        
        if model.ping > settings.criticalPing:
            score += (model.ping - settings.criticalPing) / settings.criticalPing * 5
        
        if model.jitter > settings.criticalJitter:
            score += (model.jitter - settings.criticalJitter) / settings.criticalJitter * 5
        
        return score
    
    def predict_with_fallback(self, parameters_settings: ParametersSettings,
                            models_parameters: List[ModelParameters]) -> MLPredictionResponse:
        """Прогнозирование с откатом к традиционной балансировке если ML неуверен"""
        
        ml_prediction = self.predict(parameters_settings, models_parameters)
        
        # Если ML неуверен (уверенность < 0.7), используем традиционный подход
        if ml_prediction.confidence < 0.7:
            return self._traditional_balancing(parameters_settings, models_parameters)
        
        return ml_prediction
    
    def _traditional_balancing(self, parameters_settings: ParametersSettings,
                             models_parameters: List[ModelParameters]) -> MLPredictionResponse:
        """Традиционный подход к балансировке как резерв"""
        
        # Вычисление баллов нагрузки для всех моделей
        load_scores = []
        for model in models_parameters:
            score = self._calculate_load_score(model, parameters_settings)
            load_scores.append(score)
        
        # Проверка, нужна ли балансировка какой-либо модели
        max_score = max(load_scores)
        if max_score <= 0:
            return MLPredictionResponse(isNeedIntervene=False, confidence=1.0)
        
        # Поиск наиболее и наименее загруженных моделей
        sending_idx = load_scores.index(max_score)
        receiving_idx = load_scores.index(min(load_scores))
        
        if sending_idx == receiving_idx:
            return MLPredictionResponse(isNeedIntervene=False, confidence=1.0)
        
        return MLPredictionResponse(
            isNeedIntervene=True,
            sendingModelId=models_parameters[sending_idx].id,
            acceptingModelId=models_parameters[receiving_idx].id,
            confidence=1.0
        ) 