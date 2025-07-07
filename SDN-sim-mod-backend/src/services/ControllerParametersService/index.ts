import DelayElement from "../../domains/DelayElement";
import QueueElement from "../../domains/QueueElement";
import SourceElement from "../../domains/SourceElement";
import SinkElement from "../../domains/SinkElement";
import { TLostSinkElement } from "../../domains/QueueElement/meta";
import { DEFAULT_USED_DISK_SPACE, getRandomArbitrary, DEFAULT_WORK_INTERVAL_VALUE, DEFAULT_DELAY_VALUE, MILLISECONDS_TO_SECONDS_MULTIPLIER } from "../../utils/constants";
import { TControllerParameter } from "../../domains/Controller/meta";


export const MIN_DEFAULT_MEMORY_USAGE: TControllerParameter = 0.23;
export const MAX_DEFAULT_MEMORY_USAGE: TControllerParameter = 0.27;

class ControllerParametersService {
    public static getUsedDiskSpace(queueElements: QueueElement[]): TControllerParameter {
        const usedDiskSpace = DEFAULT_USED_DISK_SPACE;

        return usedDiskSpace;
    }

    public static getMemoryUsage(queueElements: QueueElement[], delayElements: DelayElement[]): TControllerParameter {
        const defaultMemoryUsage: TControllerParameter = getRandomArbitrary(MIN_DEFAULT_MEMORY_USAGE, MAX_DEFAULT_MEMORY_USAGE);

        const queueElementsAgentsCount: TControllerParameter = queueElements.reduce((agentsCount, queueElement) => agentsCount + queueElement.getAgentsCount(), 0);
        const delayElementsAgentsCount: TControllerParameter = delayElements.reduce((agentsCount, delayElement) => agentsCount + delayElement.getAgentsCount(), 0);

        const queueElementsCapacitySum: TControllerParameter = queueElements.reduce((capacitySum, queueElement) => capacitySum + queueElement.getCapacity(), 0);
        const delayElementsCapacitySum: TControllerParameter = delayElements.reduce((capacitySum, delayElement) => capacitySum + delayElement.getCapacity(), 0);

        const agentsMemoryUsageMultiplier: TControllerParameter = 1 - defaultMemoryUsage;   
        
        return defaultMemoryUsage + (queueElementsAgentsCount + delayElementsAgentsCount) * agentsMemoryUsageMultiplier / (queueElementsCapacitySum + delayElementsCapacitySum);
    }

    public static getNetworkTraffic(sourceElements: SourceElement[]): TControllerParameter {
        return sourceElements.reduce((networkTraffic, sourceElement) => networkTraffic + sourceElement.getReceiptIntensity(), 0);
    }

    public static getPacketLost(modelSinkElement: SinkElement, queueElements: QueueElement[]): TControllerParameter {
        const lostSinkElements: TLostSinkElement[] = queueElements.map((queueElement) => queueElement.getLostSinkElement());

        const lostAgentsCount: TControllerParameter = lostSinkElements.reduce((lostAgentsCount, lostSinkElement) => {
            if (!lostSinkElement) {
                return lostAgentsCount + 0;
            }

            return lostAgentsCount + lostSinkElement.getAgentsCameCount();
        }, 0);
        const servicedAgentsCount: TControllerParameter = modelSinkElement.getAgentsCameCount();

        return servicedAgentsCount === 0 ? 0 : lostAgentsCount / servicedAgentsCount;
    }

    public static getPing(servicedSinkElement: SinkElement): TControllerParameter {
        const lastAgentCameTime: TControllerParameter = servicedSinkElement.getStatisticForController().lastAgentCameTime;
        const lastAgentLeftTime: TControllerParameter = servicedSinkElement.getStatisticForController().lastAgentLeftTime;

        return lastAgentLeftTime - lastAgentCameTime;
    }

    public static getJitter(servicedSinkElement: SinkElement): TControllerParameter {
        const previousAgentLeftTime: TControllerParameter = servicedSinkElement.getStatisticForController().previousAgentLeftTime;
        const lastAgentLeftTime: TControllerParameter = servicedSinkElement.getStatisticForController().lastAgentLeftTime;

        return lastAgentLeftTime - previousAgentLeftTime;
    }

    public static getCPU(sourceElements: SourceElement[], delayElements: DelayElement[]): TControllerParameter {
        const delaysCapacity: number = delayElements.reduce((delaysCapacity, delayElement) => delaysCapacity + delayElement.getCapacity(), 0)

        const receiptIntensity: TControllerParameter = this.getNetworkTraffic(sourceElements) / (DEFAULT_WORK_INTERVAL_VALUE / DEFAULT_DELAY_VALUE);
        
        const loadFactor: TControllerParameter = receiptIntensity * (DEFAULT_DELAY_VALUE / MILLISECONDS_TO_SECONDS_MULTIPLIER) / delaysCapacity;

        return loadFactor <= 1 ? loadFactor : 1;
    }
}

export default ControllerParametersService;