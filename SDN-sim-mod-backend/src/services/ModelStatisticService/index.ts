import { IModelStateInfo, INetworElementState, TModelStatesInfo, TStateInfo } from "../../domains/Model/meta";
import { MILLISECONDS_TO_SECONDS_MULTIPLIER } from "../../utils/constants";
import { NetworkElementsTypes, StatisticFieldsNames } from "./meta";

class ModelStatisticService {
    public static getStatisticFieldValueByFieldName(statisticFields: TStateInfo, fieldName: string): number {
        const statisticField = statisticFields.find((field) => field.fieldName === fieldName);

        if (!statisticField) {
            return 0;
        }

        return Number(statisticField.fieldValue);
    };

    public static getNetworkElementsListByName(networkElements: INetworElementState[], elementName: string): INetworElementState[] {
        const currentElementsList: INetworElementState[] = networkElements.filter((element) => element.type === elementName);

        return currentElementsList;
    }

    public static getAgentsCameInModelCount(modelCurrentState: IModelStateInfo): number {
        const networkElementList = modelCurrentState.networkElementsStatesList;

        const sourceElements: INetworElementState[] = this.getNetworkElementsListByName(networkElementList, NetworkElementsTypes.SOURCE);

        const agentsCameInModelCount = sourceElements.reduce((agentsCameInModelCount, element) => {
            const agentsCameCount = this.getStatisticFieldValueByFieldName(element.statisticFields, StatisticFieldsNames.AGENTS_LEFT_COUNT);

            return agentsCameInModelCount + agentsCameCount;
        }, 0);

        return agentsCameInModelCount;
    }

    public static getAgentsLeftThroughModelCount(modelCurrentState: IModelStateInfo): number {
        const networkElementList = modelCurrentState.networkElementsStatesList;

        const sinkElements: INetworElementState[] = this.getNetworkElementsListByName(networkElementList, NetworkElementsTypes.SINK);

        const agentsLeftThroughModelCount = sinkElements.reduce((agentsLeftThroughModelCount, element) => {
            const agentsLeftCount = this.getStatisticFieldValueByFieldName(element.statisticFields, StatisticFieldsNames.AGENTS_CAME_COUNT);

            return agentsLeftThroughModelCount + agentsLeftCount;
        }, 0);

        return agentsLeftThroughModelCount;
    }

    public static getAgentsLostInModelCount(modelCurrentState: IModelStateInfo): number {
        const networkElementList = modelCurrentState.networkElementsStatesList;

        const queueElements: INetworElementState[] = this.getNetworkElementsListByName(networkElementList, NetworkElementsTypes.QUEUE);

        const agentsLostInModelCount = queueElements.reduce((agentsLostInModelCount, element) => {
            const agentsLostCount = this.getStatisticFieldValueByFieldName(element.statisticFields, StatisticFieldsNames.AGENTS_LOST_COUNT);

            return agentsLostInModelCount + agentsLostCount;
        }, 0);

        return agentsLostInModelCount;
    }

    public static getAgentsInModelCount(modelCurrentState: IModelStateInfo): number {
        // const agentsCameInModelCount = this.getAgentsCameInModelCount(modelCurrentState);
        // const agentsLeftThroughModelCount = this.getAgentsLeftThroughModelCount(modelCurrentState);
        // const agentsLostInModelCount = this.getAgentsLostInModelCount(modelCurrentState);

        // return agentsCameInModelCount - agentsLeftThroughModelCount - agentsLostInModelCount;

        const queueElements = this.getNetworkElementsListByName(modelCurrentState.networkElementsStatesList, NetworkElementsTypes.QUEUE);
        const delayElements = this.getNetworkElementsListByName(modelCurrentState.networkElementsStatesList, NetworkElementsTypes.DELAY);

        const agentsInQueueElementsCount = queueElements.reduce((count, queueElement) => count + this.getStatisticFieldValueByFieldName(queueElement.statisticFields, StatisticFieldsNames.AGENTS_COUNT), 0);
        const agentsInDelayElementsCount = delayElements.reduce((count, delayElement) => count + this.getStatisticFieldValueByFieldName(delayElement.statisticFields, StatisticFieldsNames.AGENTS_COUNT), 0);

        return agentsInQueueElementsCount + agentsInDelayElementsCount;
    }

    public static getReceiptIntensity(modelLastState: IModelStateInfo): number {
        let receiptIntensity: number = 0;

        const networkElements = modelLastState.networkElementsStatesList;

        const sourceElments = this.getNetworkElementsListByName(networkElements, NetworkElementsTypes.SOURCE);

        sourceElments.forEach((sourceElement) => {
            receiptIntensity += this.getStatisticFieldValueByFieldName(sourceElement.statisticFields, StatisticFieldsNames.RECEIPT_INTENSITY);
        });

        return receiptIntensity;
    }

    public static getServiceIntensity(modelPreviousState: IModelStateInfo, modelLastState: IModelStateInfo): number {
        const agentsLeftThroughModelPreviousTiming: number = this.getAgentsLeftThroughModelCount(modelPreviousState);
        const agentsLeftThroughModelLastTiming: number = this.getAgentsLeftThroughModelCount(modelLastState);

        return agentsLeftThroughModelLastTiming - agentsLeftThroughModelPreviousTiming;
    }

    public static getDelayCapacity(modelLastState: IModelStateInfo) {
        const networkElementsList = modelLastState.networkElementsStatesList;

        const delayElements = this.getNetworkElementsListByName(networkElementsList, NetworkElementsTypes.DELAY);

        const delayCapacity: number = this.getStatisticFieldValueByFieldName(delayElements[0].statisticFields, StatisticFieldsNames.DELAY_CAPACITY);

        return delayCapacity;
    }

    public static getDelayValue(modelLastState: IModelStateInfo) {
        const networkElementsList = modelLastState.networkElementsStatesList;

        const delayElements = this.getNetworkElementsListByName(networkElementsList, NetworkElementsTypes.DELAY);

        const delayValue: number = this.getStatisticFieldValueByFieldName(delayElements[0].statisticFields, StatisticFieldsNames.DELAY_VALUE);

        return delayValue;
    }

    public static getLoadFactor(modelLastState: IModelStateInfo, delayValueToIntervalValueMultiplier: number): number {
        const receiptIntensity: number = this.getReceiptIntensity(modelLastState);
        const delayValue: number = this.getDelayValue(modelLastState);
        const delayCapacity: number = this.getDelayCapacity(modelLastState);

        return (receiptIntensity / delayValueToIntervalValueMultiplier) * (delayValue / MILLISECONDS_TO_SECONDS_MULTIPLIER) / delayCapacity;
    }

    public static getLoadFactorsList(modelStatesList: TModelStatesInfo, delayValueToIntervalValueMultiplier: number): number[] {
        const loadFactorList: number[] = [];

        modelStatesList.forEach((modelCurrentState: IModelStateInfo, index) => {
            const loadFactor: number = this.getLoadFactor(modelCurrentState, delayValueToIntervalValueMultiplier);

            loadFactorList.push(loadFactor);
        });

        return loadFactorList;
    }

    public static getReceiptIntensityList(modelStatesList: TModelStatesInfo): number[] {
        const receiptIntensityList: number[] = [];

        modelStatesList.forEach((modelCurrentState: IModelStateInfo, index) => {
            const receiptIntensity: number = this.getReceiptIntensity(modelCurrentState);

            receiptIntensityList.push(receiptIntensity);
        });

        return receiptIntensityList;
    }

    public static getServiceIntensityList(modelStatesList: TModelStatesInfo): number[] {
        const serviceIntensityList: number[] = [];

        modelStatesList.forEach((modelCurrentState: IModelStateInfo, index) => {
            if (!index) {
                serviceIntensityList.push(0);

                return;
            }

            const receiptIntensity: number = this.getServiceIntensity(modelStatesList[index - 1], modelCurrentState);

            serviceIntensityList.push(receiptIntensity);
        });

        return serviceIntensityList;
    };

    public static getQueueLoad(modelLastState: IModelStateInfo): number {
        let queueLoad: number = 0;

        const networkElements = modelLastState.networkElementsStatesList;
        const queueElements = networkElements.filter((element) => element.type === NetworkElementsTypes.QUEUE);

        queueElements.forEach((element) => {
            const agentsCount = this.getStatisticFieldValueByFieldName(element.statisticFields, StatisticFieldsNames.AGENTS_COUNT);

            queueLoad += agentsCount;
        });

        return queueLoad;
    }

    public static getQueueLoadList(modelStatesList: TModelStatesInfo): number[] {
        const queueLoadList: number[] = [];

        modelStatesList.forEach((modelCurrentState) => {
            const networkElements = modelCurrentState.networkElementsStatesList;

            const queueElements = networkElements.filter((element) => element.type === NetworkElementsTypes.QUEUE);

            queueElements.forEach((element) => {
                const agentsCount = this.getStatisticFieldValueByFieldName(element.statisticFields, StatisticFieldsNames.AGENTS_COUNT);

                queueLoadList.push(agentsCount)
            });
        });

        return queueLoadList;
    }
}

export default ModelStatisticService;