export const DELAY_VALUE_TO_INTERVAL_VALUE_MULTIPLIER: number = 2;

export const enum NetworkElementsTypes {
    SOURCE = "SourceElement",
    QUEUE = "QueueElement",
    DELAY = "DelayElement",
    SINK = "SinkElement",
}

export const enum StatisticFieldsNames {
    AGENTS_CAME_COUNT = "agentsCameCount",
    AGENTS_COUNT = "agentsCount",
    AGENTS_LEFT_COUNT = "agentsLeftCount",
    AGENTS_LOST_COUNT = "agentsLostCount",
    DELAY_VALUE = "delayValue",
    DELAY_CAPACITY = "delayCapacity",
    RECEIPT_INTENSITY = "receiptIntensity",
}