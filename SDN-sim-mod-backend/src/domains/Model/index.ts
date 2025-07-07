import SourceElement from "../SourceElement";
import NetworkElement from "../NetworkElement";
import QueueElement from "../QueueElement";
import DelayElement from "../DelayElement";
import SinkElement from "../SinkElement";
import Agent from "../Agent";
import { startDate } from "../..";
import { randomUUID } from "crypto";
import { ICurrentState, TBoardTime } from "../meta";
import { IModelStateInfo, IModelStatistic, INetworElementState, IStateInfoField, TModelID, TStateInfo } from "./meta";
import { getRandomArbitrary } from "../../utils/constants";
import ModelStatisticService from "../../services/ModelStatisticService";

class Model {
    private ID: TModelID;
    private sourceElements: SourceElement[];
    private networkElements: NetworkElement[];
    private queueElements: QueueElement[];
    private delayElements: DelayElement[];
    private sinkElement: SinkElement | null;
    private statistic: IModelStatistic;

    constructor() {
        this.ID = randomUUID();
        this.sourceElements = [];
        this.networkElements = [];
        this.queueElements = [];
        this.delayElements = [];
        this.sinkElement = null;
        this.statistic = {
            allAgentsCount: 0,
        };
    }

    public getID(): string {
        return this.ID;
    }

    public getSourceElements(): SourceElement[] {
        return this.sourceElements;
    }

    public getNetworkElements(): NetworkElement[] {
        return this.networkElements;
    }

    public getQueueElements(): QueueElement[] {
        return this.queueElements;
    }

    public getDelayElements(): DelayElement[] {
        return this.delayElements;
    }

    public getSinkElement(): SinkElement | null {
        return this.sinkElement;
    }

    public getStatistic(): IModelStatistic {
        return this.statistic;
    }

    public spawnAgents(minSpawnAgentsValue: number, maxSpawnAgentsValue: number ): void {
        const sourceElements = this.sourceElements;
        for (let agentIndex = 0; agentIndex < getRandomArbitrary(minSpawnAgentsValue, maxSpawnAgentsValue); agentIndex++) {
            sourceElements.forEach((element) => {
                const agent = new Agent();

                const agentId =  this.statistic.allAgentsCount + 1;
                this.statistic.allAgentsCount++;

                const agentCameTime = (new Date()).getTime() - startDate.getTime();

                agent.setId(agentId);
                agent.setCameTime(agentCameTime);

                element.trigger("system", agent);
            });
        }
    }

    public getModelStateInfo = (workTime: TBoardTime): IModelStateInfo => {
        const networkElements = this.networkElements;

        console.log("\n\n\n");
        console.log(`Model ID: ${this.ID}\n`);

        const currentState: IModelStateInfo = {
            time: String(workTime),
            networkElementsStatesList: [],
        };

        networkElements.forEach((element) => {
            const currentNetworkElementState: INetworElementState = {
                id: element.getId(),
                type: element.constructor.name,
                statisticFields: [],
            };

            const modelElementStatistic: ICurrentState = element.getCurrentState();

            console.log(`[${element.constructor.name}#${element.getId()}]`);

            const statisticFieldsArray = Object.entries(modelElementStatistic).map(([fieldName, fieldValue]) => {
                const currentStatisticField: IStateInfoField = {
                    fieldName: fieldName,
                    fieldValue: String(fieldValue),
                };

                currentNetworkElementState.statisticFields.push(currentStatisticField);

                return { Field: fieldName, Value: fieldValue };
            });

            console.table(statisticFieldsArray);

            currentState.networkElementsStatesList.push(currentNetworkElementState);
        });

        return currentState;
    }

    public getLoadFactor(workTime: TBoardTime, delayValueToIntervalValueMultiplier: number): number {
        const currentModelStateInfo = this.getModelStateInfo(workTime);

        return ModelStatisticService.getLoadFactor(currentModelStateInfo, delayValueToIntervalValueMultiplier);
    }

    public setSourceElements(sourceElements: SourceElement[]): void {
        this.sourceElements = sourceElements;
    }

    public setNetworkElements(networkElements: NetworkElement[]): void {
        this.networkElements = networkElements;
    }

    public setQueueElements(queueElements: QueueElement[]): void {
        this.queueElements = queueElements;
    }

    public setDelayElements(delayElements: DelayElement[]): void {
        this.delayElements = delayElements;
    }

    public setSinkElement(sinkElement: SinkElement): void {
        this.sinkElement = sinkElement;
    }

    public setStatistic(statistic: IModelStatistic): void {
        this.statistic = statistic;
    }

    public stop(): void {
        this.delayElements.forEach((element) => {
            element.stop();
        })
        this.queueElements.forEach((element) => {
            element.stop();
        });

        this.clearAgents();
    }

    public addSourceElement(sourceElement: SourceElement): void {
        this.networkElements.unshift(sourceElement);
        
        this.sourceElements.push(sourceElement);
    }

    public deleteSourceElement(deletedSourceElement: SourceElement): void {
        const updatedNetworkElements = this.networkElements.filter((element) => element.getId() !== deletedSourceElement.getId());

        this.setNetworkElements(updatedNetworkElements);
        
        this.sourceElements.shift();
    }

    public clearStatistic(): void {
        this.statistic = {
            allAgentsCount: 0,
        };
    }

    public clearIntervalStatistic(): void {
        this.sourceElements.forEach((sourceElement) => {
            sourceElement.clearReceiptIntensity();
        })
    }

    public clearAgents(): void {
        this.networkElements.forEach((element) => {
            element.setAgentsCameCount(0);
            element.setAgentsCount(0);
            element.setAgentsLeftCount(0);
        });

        this.queueElements.forEach((element) => {
            element.setAgentsLostCount(0);
        });
    }
}

export default Model;