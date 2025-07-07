import { BoardSettingsConfigBlocksTitles, BoardSettingsConfigFieldsInfoList, BoardSettingsConfigFieldsLabels, ISendableBoardSettingsConfig, ModelAdditionalInfoList, TBoardSettingsConfigRanges } from "../../domains/Board/meta";
import { ISendedModelAdditionalInfo } from "../../domains/Model/meta";
import { DEFAULT_DELAY_VALUE, DEFAULT_IS_QUALITY_OF_SERVICE_ACTIVE, DEFAULT_IS_ML_BALANCING_ACTIVE, DEFAULT_JITTER_DANGER_VALUE, DEFAULT_LOAD_FACTOR_DANGER_VALUE, DEFAULT_MAX_DELAY_CAPACITY, DEFAULT_MAX_QUEUE_CAPACITY, DEFAULT_MAX_SPAWN_AGENTS_VALUE, DEFAULT_MIN_DELAY_CAPACITY, DEFAULT_MIN_QUEUE_CAPACITY, DEFAULT_MIN_SPAWN_AGENTS_VALUE, DEFAULT_MODEL_SOURCE_ELEMENTS_COUNT_VALUE, DEFAULT_MODELS_COUNT_VALUE, DEFAULT_PACKET_LOST_DANGER_VALUE, DEFAULT_PING_DANGER_VALUE, DEFAULT_STATISTIC_INTERVAL_VALUE, DEFAULT_WORK_INTERVAL_VALUE } from "../../utils/constants";

class BoardSettingsConfigService {
    public static getDefaultBoardSettingsConfig(): ISendableBoardSettingsConfig {
        return {
            modelsSettings: {
                title: BoardSettingsConfigBlocksTitles.BOARD_SETTINGS,
                fields: {
                    modelsCountValue: {
                        label: BoardSettingsConfigFieldsLabels.MODELS_COUNT_VALUE,
                        value: DEFAULT_MODELS_COUNT_VALUE,
                        info: BoardSettingsConfigFieldsInfoList.MODELS_COUNT_VALUE,
                    },
                    minSpawnAgentsValue: {
                        label: BoardSettingsConfigFieldsLabels.MIN_SPAWN_AGENTS_VALUE,
                        value: DEFAULT_MIN_SPAWN_AGENTS_VALUE,
                        info: BoardSettingsConfigFieldsInfoList.MIN_SPAWN_AGENTS_VALUE,  // Add info field
                    },
                    maxSpawnAgentsValue: {
                        label: BoardSettingsConfigFieldsLabels.MAX_SPAWN_AGENTS_VALUE,
                        value: DEFAULT_MAX_SPAWN_AGENTS_VALUE,
                        info: BoardSettingsConfigFieldsInfoList.MAX_SPAWN_AGENTS_VALUE,  // Add info field
                    },
                    workIntervalValue: {
                        label: BoardSettingsConfigFieldsLabels.WORK_INTERVAL_VALUE,
                        value: DEFAULT_WORK_INTERVAL_VALUE,
                        info: BoardSettingsConfigFieldsInfoList.WORK_INTERVAL_VALUE,  // Add info field
                    },
                    statisticIntervalValue: {
                        label: BoardSettingsConfigFieldsLabels.STATISTIC_INTERVAL_VALUE,
                        value: DEFAULT_STATISTIC_INTERVAL_VALUE,
                        info: BoardSettingsConfigFieldsInfoList.STATISTIC_INTERVAL_VALUE,  // Add info field
                    },
                    modelSourceElementsCountValue: {
                        label: BoardSettingsConfigFieldsLabels.MODEL_SOURCE_ELEMENTS_COUNT_VALUE,
                        value: DEFAULT_MODEL_SOURCE_ELEMENTS_COUNT_VALUE,
                        info: BoardSettingsConfigFieldsInfoList.MODEL_SOURCE_ELEMENTS_COUNT_VALUE,  // Add info field
                    },
                    minQueueCapacity: {
                        label: BoardSettingsConfigFieldsLabels.MIN_QUEUE_CAPACITY,
                        value: DEFAULT_MIN_QUEUE_CAPACITY,
                        info: BoardSettingsConfigFieldsInfoList.MIN_QUEUE_CAPACITY,  // Add info field
                    },
                    maxQueueCapacity: {
                        label: BoardSettingsConfigFieldsLabels.MAX_QUEUE_CAPACITY,
                        value: DEFAULT_MAX_QUEUE_CAPACITY,
                        info: BoardSettingsConfigFieldsInfoList.MAX_QUEUE_CAPACITY,  // Add info field
                    },
                    minDelayCapacity: {
                        label: BoardSettingsConfigFieldsLabels.MIN_DELAY_CAPACITY,
                        value: DEFAULT_MIN_DELAY_CAPACITY,
                        info: BoardSettingsConfigFieldsInfoList.MIN_DELAY_CAPACITY,  // Add info field
                    },
                    maxDelayCapacity: {
                        label: BoardSettingsConfigFieldsLabels.MAX_DELAY_CAPACITY,
                        value: DEFAULT_MAX_DELAY_CAPACITY,
                        info: BoardSettingsConfigFieldsInfoList.MAX_DELAY_CAPACITY,  // Add info field
                    },
                    delayValue: {
                        label: BoardSettingsConfigFieldsLabels.DELAY_VALUE,
                        value: DEFAULT_DELAY_VALUE,
                        info: BoardSettingsConfigFieldsInfoList.DELAY_VALUE,
                    },
                    loadFactorDangerValue: {

                        label: BoardSettingsConfigFieldsLabels.LOAD_FACTOR_DANGER_VALUE,
                        value: DEFAULT_LOAD_FACTOR_DANGER_VALUE,
                        info: BoardSettingsConfigFieldsInfoList.LOAD_FACTOR_DANGER_VALUE,
                    },
                },
            },
            qualityOfServiceSettings: {
                title: BoardSettingsConfigBlocksTitles.QUALITY_OF_SERVICE_ACTIVE,
                isActive: DEFAULT_IS_QUALITY_OF_SERVICE_ACTIVE,
                fields: {
                    packetLostDangerValue: {
                        label: BoardSettingsConfigFieldsLabels.PACKET_LOST_DANGER_VALUE,
                        value: DEFAULT_PACKET_LOST_DANGER_VALUE,
                        info: BoardSettingsConfigFieldsInfoList.PACKET_LOST_DANGER_VALUE,
                    },
                    pingDangerValue: {
                        label: BoardSettingsConfigFieldsLabels.PING_DANGER_VALUE,
                        value: DEFAULT_PING_DANGER_VALUE,
                        info: BoardSettingsConfigFieldsInfoList.PING_DANGER_VALUE,
                    },
                    jitterDangerValue: {
                        label: BoardSettingsConfigFieldsLabels.JITTER_DANGER_VALUE,
                        value: DEFAULT_JITTER_DANGER_VALUE,
                        info: BoardSettingsConfigFieldsInfoList.JITTER_DANGER_VALUE,
                    },
                },
            },
            mlBalancingSettings: {
                title: BoardSettingsConfigBlocksTitles.ML_BALANCING_ACTIVE,
                isActive: DEFAULT_IS_ML_BALANCING_ACTIVE,
                fields: {},
            },
        };

    }

    public static getBoardSettingsConfigRanges(): TBoardSettingsConfigRanges {
        return {
            modelsCountValue: {
                minValue: 1,
                maxValue: 10,
                step: 1,
                initialValue: DEFAULT_MODELS_COUNT_VALUE,
            },
            minSpawnAgentsValue: {
                minValue: 1,
                maxValue: 20,
                step: 1,
                initialValue: DEFAULT_MIN_SPAWN_AGENTS_VALUE,
            },
            maxSpawnAgentsValue: {
                minValue: 5,
                maxValue: 100,
                step: 1,
                initialValue: DEFAULT_MAX_SPAWN_AGENTS_VALUE,
            },
            workIntervalValue: {
                minValue: 100,
                maxValue: 10000,
                step: 100,
                initialValue: DEFAULT_WORK_INTERVAL_VALUE,
            },
            statisticIntervalValue: {
                minValue: 100,
                maxValue: 10000,
                step: 100,
                initialValue: DEFAULT_STATISTIC_INTERVAL_VALUE,
            },
            modelSourceElementsCountValue: {
                minValue: 1,
                maxValue: 10,
                step: 1,
                initialValue: DEFAULT_MODEL_SOURCE_ELEMENTS_COUNT_VALUE,
            },
            minQueueCapacity: {
                minValue: 1,
                maxValue: 25,
                step: 1,
                initialValue: DEFAULT_MIN_QUEUE_CAPACITY,
            },
            maxQueueCapacity: {
                minValue: 5,
                maxValue: 100,
                step: 1,
                initialValue: DEFAULT_MAX_QUEUE_CAPACITY,
            },
            minDelayCapacity: {
                minValue: 1,
                maxValue: 25,
                step: 1,
                initialValue: DEFAULT_MIN_DELAY_CAPACITY,
            },
            maxDelayCapacity: {
                minValue: 5,
                maxValue: 100,
                step: 1,
                initialValue: DEFAULT_MAX_DELAY_CAPACITY,
            },
            delayValue: {
                minValue: 100,
                maxValue: 10000,
                step: 100,
                initialValue: DEFAULT_DELAY_VALUE,
            },
            loadFactorDangerValue: {
                minValue: 0.5,
                maxValue: 1.5,
                step: 0.1,
                initialValue: DEFAULT_LOAD_FACTOR_DANGER_VALUE,
            },
            packetLostDangerValue: {
                minValue: 0.1,
                maxValue: 1,
                step: 0.01,
                initialValue: DEFAULT_PACKET_LOST_DANGER_VALUE,
            },
            pingDangerValue: {
                minValue: 100,
                maxValue: 10000,
                step: 100,
                initialValue: DEFAULT_PING_DANGER_VALUE,
            },
            jitterDangerValue: {
                minValue: 10,
                maxValue: 200,
                step: 10,
                initialValue: DEFAULT_JITTER_DANGER_VALUE,
            },
        };
    }
    
    public static getModelsAdditionalInfo(agentsCameInModelCount: number, agentsLeftThroughModelCount: number, agentsInModelCount: number, agentsLostCount: number): ISendedModelAdditionalInfo {
        return {
            agentsCameInModelCount: {
                value: String(agentsCameInModelCount),
                info: ModelAdditionalInfoList.AGENTS_CAME_IN_MODEL,
            },
            agentsLeftThroughModelCount: {
                value: String(agentsLeftThroughModelCount),
                info: ModelAdditionalInfoList.AGENTS_LEFT_THROUGH_MODEL,
            },
            agentsInModelCount: {
                value: String(agentsInModelCount),
                info: ModelAdditionalInfoList.AGENTS_IN_MODEL,
            },
            agentsLostCount: {
                value: String(agentsLostCount),
                info: ModelAdditionalInfoList.AGENTS_LOST,
            },
        }
    }

}

export default BoardSettingsConfigService;