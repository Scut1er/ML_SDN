import Balancer from "../Balancer";
import MLBalancer from "../MLBalancer";
import Controller from "../Controller";
import { TParametersStatesList } from "../Controller/meta";

export type TControllersList = Controller[];
export type TControllersStatesList = TParametersStatesList[];

export type TBoardBalancer = Balancer | MLBalancer | null;

export type TModelsInterval = NodeJS.Timer | null;

export interface ISettingsConfig {
    modelsCountValue: number;
    minSpawnAgentsValue: number;
    maxSpawnAgentsValue: number;
    workIntervalValue: number;
    statisticIntervalValue: number;
    modelSourceElementsCountValue: number;
    minQueueCapacity: number;
    maxQueueCapacity: number;
    minDelayCapacity: number;
    maxDelayCapacity: number;
    delayValue: number;
    loadFactorDangerValue: number;
    packetLostDangerValue: number;
    pingDangerValue: number;
    jitterDangerValue: number;
    isQualityOfServiceActive: boolean;
    isMLBalancingActive: boolean;
}

export interface ISendableSettingsConfigField {
    label: string;
    value: number;
    info: string;
}

export interface ISendableBoardSettingsConfigBlock {
    isActive?: boolean;
    title: string;
    fields: { [key: string]: ISendableSettingsConfigField };
}

export interface ISendableBoardSettingsConfig {
    modelsSettings: ISendableBoardSettingsConfigBlock;
    qualityOfServiceSettings: ISendableBoardSettingsConfigBlock;
    mlBalancingSettings: ISendableBoardSettingsConfigBlock;
}

export const enum BoardSettingsConfigFieldsLabels {
    MODELS_COUNT_VALUE = "Models",
    MIN_SPAWN_AGENTS_VALUE = "Minimum Agents per tick",
    MAX_SPAWN_AGENTS_VALUE = "Maximum Agents per tick",
    WORK_INTERVAL_VALUE = "Interval between ticks, ms",
    STATISTIC_INTERVAL_VALUE = "Polling frequency, ms",
    MODEL_SOURCE_ELEMENTS_COUNT_VALUE = "Agent Sources per Model",
    MIN_QUEUE_CAPACITY = "Minimum Queue capacity",
    MAX_QUEUE_CAPACITY = "Maximum Queue capacity",
    MIN_DELAY_CAPACITY = "Minimum Server capacity",
    MAX_DELAY_CAPACITY = "Maximum Server capacity",
    DELAY_VALUE = "Agent processing time, ms",
    LOAD_FACTOR_DANGER_VALUE = "Load factor boundary value",
    PACKET_LOST_DANGER_VALUE = "Agents loss boundary value",
    PING_DANGER_VALUE = "Ping boundary value, ms",
    JITTER_DANGER_VALUE = "Jitter boundary value, ms",
}

export const enum BoardSettingsConfigFieldsInfoList {
    MODELS_COUNT_VALUE = "Sets the number of models in the experiment",
    MIN_SPAWN_AGENTS_VALUE = "Sets the minimum number of Agents arriving per tick",
    MAX_SPAWN_AGENTS_VALUE = "Sets the maximum number of Agents arriving per tick",
    WORK_INTERVAL_VALUE = "Sets the time interval between ticks",
    STATISTIC_INTERVAL_VALUE = "Sets the time interval between statistics collection",
    MODEL_SOURCE_ELEMENTS_COUNT_VALUE = "Sets the number of Agent Sources",
    MIN_QUEUE_CAPACITY = "Sets the minimum Queue capacity",
    MAX_QUEUE_CAPACITY = "Sets the maximum Queue capacity",
    MIN_DELAY_CAPACITY = "Sets the minimum Server capacity",
    MAX_DELAY_CAPACITY = "Sets the maximum Server capacity",
    DELAY_VALUE = "Sets the time it takes for one Agent to be processed",
    LOAD_FACTOR_DANGER_VALUE = "Sets the boundary value of the LoadFactor at which the load balancing decision will be made",
    PACKET_LOST_DANGER_VALUE = "Sets the AgentsLoss boundary value at which a load balancing decision will be made",
    PING_DANGER_VALUE = "Sets the Ping boundary value at which the load balancing decision will be made",
    JITTER_DANGER_VALUE = "Sets the Jitter boundary value at which the load balancing decision will be made",
}

export const enum BoardSettingsConfigBlocksTitles {
    BOARD_SETTINGS = "Board settings",
    QUALITY_OF_SERVICE_ACTIVE = "QOS settings",
    ML_BALANCING_ACTIVE = "ML Balancing",
}

export interface IRangeSettingData {
    minValue: number;
    maxValue: number;
    step: number;
    initialValue: number;
};

export type TBoardSettingsConfigRanges = Record<string, IRangeSettingData>;

export interface IModelRating {
    currentValue: number;
    maximumValue: number;
    info: string;
}

export interface IModelRatingInfo {
    queue: IModelRating;
    delay: IModelRating;
    general: IModelRating;
}

export type TModelsRatings = IModelRatingInfo[];

export const enum ModelRatingInfoList {
    QUEUE = "Shows the ratio of Queue capacity to the maximum possible Queue capacity in this Model",
    DELAY = "Shows the ratio of the Server capacity to the maximum possible Server capacity in this Model",
    GENERAL = "Shows the overall estimate of the Model's capacity relative to the maximum possible capacity of the Model in the experiment",
}

export const enum ModelAdditionalInfoList {
    AGENTS_CAME_IN_MODEL = "Shows how many Agents entered the Model from all Sources for the entire work time of the Model",
    AGENTS_LEFT_THROUGH_MODEL = "Shows how many Agents left the Model from all Sources for the entire work time of the Model",
    AGENTS_IN_MODEL = "Shows how many Agents are in the Model for at the moment",
    AGENTS_LOST = "Shows how many Agents could not be processed and were lost",
}