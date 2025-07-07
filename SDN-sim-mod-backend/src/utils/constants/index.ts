import NetworkElement from "../../domains/NetworkElement";
import { TPreviousNetworkElements } from "../../domains/meta";

export const DEFAULT_MODELS_COUNT_VALUE: number = 10;

export const DEFAULT_MIN_SPAWN_AGENTS_VALUE: number = 3;
export const DEFAULT_MAX_SPAWN_AGENTS_VALUE: number = 6;
export const DEFAULT_WORK_INTERVAL_VALUE: number = 2000;
export const DEFAULT_STATISTIC_INTERVAL_VALUE: number = 2000;
export const DEFAULT_MODEL_SOURCE_ELEMENTS_COUNT_VALUE: number = 3;
export const DEFAULT_MIN_QUEUE_CAPACITY: number = 5;
export const DEFAULT_MAX_QUEUE_CAPACITY: number = 15;
export const DEFAULT_MIN_DELAY_CAPACITY: number = 5;
export const DEFAULT_MAX_DELAY_CAPACITY: number = 40;

export const DEFAULT_DELAY_VALUE: number = 1000;
export const DEFAULT_IS_QUALITY_OF_SERVICE_ACTIVE: boolean = false;
export const DEFAULT_IS_ML_BALANCING_ACTIVE: boolean = false;

export const DEFAULT_LOAD_FACTOR_DANGER_VALUE: number = 1;
export const DEFAULT_PACKET_LOST_DANGER_VALUE: number = 0.9;
export const DEFAULT_PING_DANGER_VALUE: number = 2000;
export const DEFAULT_JITTER_DANGER_VALUE: number = 250;

export const PORT: number = 5500;
export const EXPRESS_APP_ALLOWED_CORS_URL: string = "http://localhost:5173";

export const DEFAULT_USED_DISK_SPACE: number = 0.4;

export const WEB_CLIENT_PORT: number = 3001;

export const MILLISECONDS_TO_SECONDS_MULTIPLIER: number = 1000;

export const getRandomArbitrary = (min: number, max: number) => {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

export const addElementsInList = (list: NetworkElement[], ...elements: NetworkElement[]): void => {
    elements.forEach((element) => {
        list.push(element);
    })
}

export const getPreviousElementsList = (...elements: NetworkElement[]): TPreviousNetworkElements => {
    const previousElements: TPreviousNetworkElements = new Map<string, NetworkElement>;

    elements.forEach((element) => {
        previousElements.set(element.getId(), element);
    })

    return previousElements;
}

export const settingNextElementsInSequence = (elements: NetworkElement[]): void => {
    const lastElementIndex = elements.length - 1;

    elements.forEach((element, index) => {
        if (index == lastElementIndex) {
            return;
        }

        element.setNextElement(elements[index + 1]);
    })
}

export const combineArray = (firstArray:any[], secondArray: any[]): void => {
    secondArray.forEach((element) => {
        firstArray.push(element);
    });
}

export const getMaxElementIndex = (list: number[]): number => {
    let maxValue: number = list[0];
    let maxElementIndex: number = 0;

    for (let index = 1; index < list.length; index++) {
        if (list[index] > maxValue) {
            maxElementIndex = index;
            maxValue = list[index];
        }
    }

    return maxElementIndex;
}

export const getMinElementIndex = (list: number[]): number => {
    let minValue: number = list[0];
    let minElementIndex: number = 0;

    for (let index = 1; index < list.length; index++) {
        if (list[index] < minValue) {
            minElementIndex = index;
            minValue = list[index];
        }
    }

    return minElementIndex;
}