import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Tuple, Dict, Any
from models.neural_network import LoadBalancerNN
from models.types import NetworkSolution, ModelParameters, ParametersSettings, BalancingSolution

class BalancingDataset(Dataset):
    def __init__(self, data: List[NetworkSolution]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        solution = self.data[idx]
        
        # Prepare features (same as in neural_network.py)
        features = self._prepare_features(solution.parametersSettings, solution.modelsParameters)
        
        # Prepare targets
        targets = self._prepare_targets(solution)
        
        return features, targets
    
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
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _prepare_targets(self, solution: NetworkSolution) -> Dict[str, torch.Tensor]:
        """Подготовка целевых значений для обучения"""
        
        # Используем firstWaySolution (LoadFactor) как истинное значение
        # Можно изменить на secondWaySolution (QoS) или комбинированный
        target_solution = solution.firstWaySolution
        
        # Цель решения (бинарная: нужна балансировка или нет)
        decision_target = torch.tensor(1.0 if target_solution.isNeedIntervene else 0.0, 
                                     dtype=torch.float32)
        
        # Цели выбора модели
        sending_target = torch.zeros(10, dtype=torch.float32)  # Максимум 10 моделей
        receiving_target = torch.zeros(10, dtype=torch.float32)
        
        if target_solution.isNeedIntervene and target_solution.sendingModelId and target_solution.acceptingModelId:
            # Найти индексы отправляющей и принимающей модели
            model_ids = [model.id for model in solution.modelsParameters]
            
            try:
                sending_idx = model_ids.index(target_solution.sendingModelId)
                receiving_idx = model_ids.index(target_solution.acceptingModelId)
                
                sending_target[sending_idx] = 1.0
                receiving_target[receiving_idx] = 1.0
            except ValueError:
                # ID модели не найден, используем нулевые цели
                pass
        
        return {
            'decision': decision_target,
            'sending': sending_target,
            'receiving': receiving_target
        }

class TrainingService:
    def __init__(self, model_save_path: str = "models/trained_model.pth"):
        self.model = LoadBalancerNN()
        self.model_save_path = model_save_path
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Функции потерь
        self.decision_loss_fn = nn.BCELoss()
        self.selection_loss_fn = nn.CrossEntropyLoss()
    
    def load_data_from_json(self, json_file_path: str) -> List[NetworkSolution]:
        """Загрузка обучающих данных из JSON файла"""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        solutions = []
        for item in data:
            # Парсинг NetworkSolution из словаря
            solution = self._parse_network_solution(item)
            solutions.append(solution)
        
        return solutions
    
    def _parse_network_solution(self, data: Dict[str, Any]) -> NetworkSolution:
        """Парсинг NetworkSolution из словаря"""
        
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
        
        # Парсинг решений
        first_way = BalancingSolution(
            isNeedIntervene=data['firstWaySolution']['isNeedIntervene'],
            sendingModelId=data['firstWaySolution'].get('sendingModelId'),
            acceptingModelId=data['firstWaySolution'].get('acceptingModelId')
        )
        
        second_way = BalancingSolution(
            isNeedIntervene=data['secondWaySolution']['isNeedIntervene'],
            sendingModelId=data['secondWaySolution'].get('sendingModelId'),
            acceptingModelId=data['secondWaySolution'].get('acceptingModelId')
        )
        
        return NetworkSolution(
            parametersSettings=params_settings,
            modelsParameters=models_params,
            firstWaySolution=first_way,
            secondWaySolution=second_way
        )
    
    def train(self, training_data: List[NetworkSolution], epochs: int = 50, 
              batch_size: int = 32, validation_split: float = 0.2) -> Dict[str, List[float]]:
        """Обучение нейронной сети"""
        
        # Разделение данных
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        # Создание датасетов и загрузчиков данных
        train_dataset = BalancingDataset(train_data)
        val_dataset = BalancingDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # История обучения
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        self.model.train()
        
        for epoch in range(epochs):
            # Фаза обучения
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Фаза валидации
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Обновление скорости обучения
            self.scheduler.step()
            
            # Запись истории
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            
            print(f"Эпоха {epoch+1}/{epochs}: "
                  f"Потери обучения: {train_loss:.4f}, Точность обучения: {train_acc:.4f}, "
                  f"Потери валидации: {val_loss:.4f}, Точность валидации: {val_acc:.4f}")
        
        # Сохранение обученной модели
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Модель сохранена в {self.model_save_path}")
        
        return history
    
    def _train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Обучение одной эпохи"""
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for features, targets in dataloader:
            self.optimizer.zero_grad()
            
            # Прямой проход
            decision, sending_probs, receiving_probs = self.model(features)
            
            # Вычисление потерь
            decision_loss = self.decision_loss_fn(decision.squeeze(), targets['decision'].squeeze())
            
            # Для многоклассовых потерь вычисляем только когда есть положительное решение
            sending_loss = torch.tensor(0.0)
            receiving_loss = torch.tensor(0.0)
            
            positive_decisions = targets['decision'].squeeze() > 0.5
            if positive_decisions.any():
                sending_targets = torch.argmax(targets['sending'][positive_decisions], dim=1)
                receiving_targets = torch.argmax(targets['receiving'][positive_decisions], dim=1)
                
                if len(sending_targets) > 0:
                    sending_loss = self.selection_loss_fn(
                        sending_probs[positive_decisions], sending_targets
                    )
                    receiving_loss = self.selection_loss_fn(
                        receiving_probs[positive_decisions], receiving_targets
                    )
            
            # Общие потери
            total_loss_batch = decision_loss + sending_loss + receiving_loss
            
            # Обратный проход
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Накопление метрик
            total_loss += total_loss_batch.item()
            
            # Вычисление точности (для головы решения)
            predictions = (decision.squeeze() > 0.5).float()
            decision_targets = targets['decision'].squeeze()
            correct_predictions += (predictions == decision_targets).sum().item()
            total_predictions += decision_targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Валидация одной эпохи"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for features, targets in dataloader:
                # Прямой проход
                decision, sending_probs, receiving_probs = self.model(features)
                
                # Вычисление потерь
                decision_loss = self.decision_loss_fn(decision.squeeze(), targets['decision'].squeeze())
                
                sending_loss = torch.tensor(0.0)
                receiving_loss = torch.tensor(0.0)
                
                positive_decisions = targets['decision'].squeeze() > 0.5
                if positive_decisions.any():
                    sending_targets = torch.argmax(targets['sending'][positive_decisions], dim=1)
                    receiving_targets = torch.argmax(targets['receiving'][positive_decisions], dim=1)
                    
                    if len(sending_targets) > 0:
                        sending_loss = self.selection_loss_fn(
                            sending_probs[positive_decisions], sending_targets
                        )
                        receiving_loss = self.selection_loss_fn(
                            receiving_probs[positive_decisions], receiving_targets
                        )
                
                # Общие потери
                total_loss_batch = decision_loss + sending_loss + receiving_loss
                
                # Накопление метрик
                total_loss += total_loss_batch.item()
                
                # Вычисление точности
                predictions = (decision.squeeze() > 0.5).float()
                decision_targets = targets['decision'].squeeze()
                correct_predictions += (predictions == decision_targets).sum().item()
                total_predictions += decision_targets.size(0)
        
        self.model.train()
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> List[NetworkSolution]:
        """Генерация синтетических обучающих данных"""
        solutions = []
        
        for _ in range(num_samples):
            # Случайные настройки параметров
            params_settings = ParametersSettings(
                criticalLoadFactor=np.random.uniform(0.7, 1.0),
                criticalPacketLost=np.random.uniform(0.05, 0.2),
                criticalPing=np.random.uniform(100, 500),
                criticalJitter=np.random.uniform(10, 50)
            )
            
            # Случайное количество моделей (2-4)
            num_models = np.random.randint(2, 5)
            models_params = []
            
            for i in range(num_models):
                model = ModelParameters(
                    id=f"model_{i+1}",
                    loadFactor=np.random.uniform(0.1, 1.2),
                    packetLost=np.random.uniform(0.0, 0.3),
                    ping=np.random.uniform(10, 600),
                    jitter=np.random.uniform(1, 60),
                    CPU=np.random.uniform(0.1, 1.0),
                    usedDiskSpace=np.random.uniform(0.1, 1.0),
                    memoryUsage=np.random.uniform(0.1, 1.0),
                    networkTraffic=np.random.uniform(0.1, 1.0)
                )
                models_params.append(model)
            
            # Генерация решений на основе правил
            first_way = self._generate_load_factor_solution(models_params, params_settings)
            second_way = self._generate_qos_solution(models_params, params_settings)
            
            solution = NetworkSolution(
                parametersSettings=params_settings,
                modelsParameters=models_params,
                firstWaySolution=first_way,
                secondWaySolution=second_way
            )
            
            solutions.append(solution)
        
        return solutions
    
    def _generate_load_factor_solution(self, models: List[ModelParameters], 
                                     settings: ParametersSettings) -> BalancingSolution:
        """Генерация решения на основе коэффициента нагрузки"""
        
        # Проверка, превышает ли какая-либо модель критический коэффициент нагрузки
        overloaded_models = [m for m in models if m.loadFactor > settings.criticalLoadFactor]
        
        if not overloaded_models:
            return BalancingSolution(isNeedIntervene=False)
        
        # Найти наиболее перегруженную модель
        most_loaded = max(models, key=lambda m: m.loadFactor)
        
        # Найти лучшего кандидата для приема нагрузки (низкая нагрузка + хорошая производительность)
        def receiving_suitability(model):
            if model.id == most_loaded.id:
                return float('inf')  # Нельзя отправлять в ту же модель
            
            # Предпочитаем модели с низкой нагрузкой и хорошей производительностью
            load_penalty = model.loadFactor
            
            # Учитываем QoS - не отправляем в уже испытывающие трудности модели
            qos_penalty = 0
            if model.packetLost > settings.criticalPacketLost * 0.7:  # 70% от критического
                qos_penalty += 0.5
            if model.ping > settings.criticalPing * 0.7:
                qos_penalty += 0.5
            if model.jitter > settings.criticalJitter * 0.7:
                qos_penalty += 0.5
            
            return load_penalty + qos_penalty
        
        best_receiver = min(models, key=receiving_suitability)
        
        if most_loaded.id == best_receiver.id:
            return BalancingSolution(isNeedIntervene=False)
        
        # Вмешиваемся только если принимающая модель может справиться с большей нагрузкой
        if best_receiver.loadFactor > settings.criticalLoadFactor * 0.8:  # 80% от критического
            return BalancingSolution(isNeedIntervene=False)
        
        # Вмешиваемся только при значительной разнице в нагрузке
        if most_loaded.loadFactor - best_receiver.loadFactor < 0.3:
            return BalancingSolution(isNeedIntervene=False)
        
        return BalancingSolution(
            isNeedIntervene=True,
            sendingModelId=most_loaded.id,
            acceptingModelId=best_receiver.id
        )
    
    def _generate_qos_solution(self, models: List[ModelParameters], 
                             settings: ParametersSettings) -> BalancingSolution:
        """Генерация решения на основе QoS"""
        
        # Проверка, есть ли у какой-либо модели проблемы с QoS
        problematic_models = [
            m for m in models 
            if (m.packetLost > settings.criticalPacketLost or 
                m.ping > settings.criticalPing or 
                m.jitter > settings.criticalJitter)
        ]
        
        if not problematic_models:
            return BalancingSolution(isNeedIntervene=False)
        
        # Найти наиболее проблемные и лучше работающие модели
        def qos_score(model):
            score = 0
            if model.packetLost > settings.criticalPacketLost:
                score += model.packetLost - settings.criticalPacketLost
            if model.ping > settings.criticalPing:
                score += (model.ping - settings.criticalPing) / settings.criticalPing
            if model.jitter > settings.criticalJitter:
                score += (model.jitter - settings.criticalJitter) / settings.criticalJitter
            return score
        
        # Комбинированный балл: проблемы QoS + коэффициент нагрузки
        def combined_score(model):
            qos = qos_score(model)
            load_penalty = max(0, model.loadFactor - settings.criticalLoadFactor) * 2
            return qos + load_penalty
        
        # Найти худшую и лучшую модели по комбинированному баллу
        worst_model = max(models, key=combined_score)
        
        # Для принимающей модели предпочитаем низкую нагрузку И хороший QoS
        def receiving_suitability(model):
            if model.id == worst_model.id:
                return float('inf')  # Нельзя отправлять в ту же модель
            
            # Предпочитаем модели с низкой нагрузкой и хорошим QoS
            qos_penalty = qos_score(model)
            load_penalty = model.loadFactor
            
            return qos_penalty + load_penalty
        
        best_model = min(models, key=receiving_suitability)
        
        if worst_model.id == best_model.id:
            return BalancingSolution(isNeedIntervene=False)
        
        # Вмешиваемся только если это имеет смысл (отправка от худшего к лучшему)
        if combined_score(worst_model) <= combined_score(best_model):
            return BalancingSolution(isNeedIntervene=False)
        
        return BalancingSolution(
            isNeedIntervene=True,
            sendingModelId=worst_model.id,
            acceptingModelId=best_model.id
        ) 