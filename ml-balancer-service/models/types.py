from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ModelParameters:
    id: str
    loadFactor: float
    packetLost: float
    ping: float
    jitter: float
    CPU: float = 0.0
    usedDiskSpace: float = 0.0
    memoryUsage: float = 0.0
    networkTraffic: float = 0.0

@dataclass
class ParametersSettings:
    criticalLoadFactor: float
    criticalPacketLost: float
    criticalPing: float
    criticalJitter: float

@dataclass
class BalancingSolution:
    isNeedIntervene: bool
    sendingModelId: Optional[str] = None
    acceptingModelId: Optional[str] = None

@dataclass
class NetworkSolution:
    parametersSettings: ParametersSettings
    modelsParameters: List[ModelParameters]
    firstWaySolution: BalancingSolution  # На основе коэффициента нагрузки
    secondWaySolution: BalancingSolution  # На основе QoS

@dataclass
class MLPredictionRequest:
    parametersSettings: ParametersSettings
    modelsParameters: List[ModelParameters]

@dataclass
class MLPredictionResponse:
    isNeedIntervene: bool
    sendingModelId: Optional[str] = None
    acceptingModelId: Optional[str] = None
    confidence: float = 0.0 