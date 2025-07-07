import Model from "../Model";
import NetworkElement from "../NetworkElement";
import SourceElement from "../SourceElement";
import QueueElement from "../QueueElement";
import DelayElement from "../DelayElement";
import SinkElement from "../SinkElement";
import Controller from "../Controller";
import Balancer from "../Balancer";
import MLBalancer from "../MLBalancer";
import ModelStatisticService from "../../services/ModelStatisticService";
import BoardSettingsConfigService from "../../services/BoardSettingsConfigService";
import { addElementsInList, DEFAULT_DELAY_VALUE, DEFAULT_IS_QUALITY_OF_SERVICE_ACTIVE, DEFAULT_JITTER_DANGER_VALUE, DEFAULT_LOAD_FACTOR_DANGER_VALUE, DEFAULT_MAX_DELAY_CAPACITY, DEFAULT_MAX_QUEUE_CAPACITY, DEFAULT_MAX_SPAWN_AGENTS_VALUE, DEFAULT_MIN_DELAY_CAPACITY, DEFAULT_MIN_QUEUE_CAPACITY, DEFAULT_MIN_SPAWN_AGENTS_VALUE, DEFAULT_MODEL_SOURCE_ELEMENTS_COUNT_VALUE, DEFAULT_MODELS_COUNT_VALUE, DEFAULT_PACKET_LOST_DANGER_VALUE, DEFAULT_PING_DANGER_VALUE, DEFAULT_STATISTIC_INTERVAL_VALUE, DEFAULT_WORK_INTERVAL_VALUE, getPreviousElementsList, getRandomArbitrary, MILLISECONDS_TO_SECONDS_MULTIPLIER, settingNextElementsInSequence } from "../../utils/constants";
import { TModelsRatings, ISendableBoardSettingsConfig, ISettingsConfig, TBoardBalancer, TControllersList, TModelsInterval, ModelRatingInfoList, IModelRatingInfo } from "./meta";
import { TControllersStatesList } from "./meta";
import { TModelsList, TBoardTime } from "../meta";
import { IModelStateInfo, TModelID, TSendedModelsAdditionalInfoList, TSendedChartsDataList } from "../Model/meta";
import { ServerMessageTypes } from "../../controllers/WebSocketController/meta";

class Board {
    private modelsList: TModelsList;
    private controllersList: TControllersList;
    private balancer: TBoardBalancer;
    private workTime: TBoardTime;
    private statisticTime: TBoardTime;
    private modelsWorkTimer: TModelsInterval;
    private sendModelsStatisticTimer: TModelsInterval;
    private isModelsCreate: boolean;
    private isModelsStart: boolean;
    private isModelsStop: boolean;
    private sendingData: TControllersStatesList;
    private sendFunction: any;
    private settingsConfig: ISettingsConfig;

    constructor() {
        this.modelsList = [];
        this.controllersList = [];
        this.balancer = null;
        this.workTime = 0;
        this.statisticTime = 0;
        this.modelsWorkTimer = null;
        this.sendModelsStatisticTimer = null;
        this.isModelsCreate = false
        this.isModelsStart = true;
        this.isModelsStop = true;
        this.sendingData = [];
        this.sendFunction = null;
        this.settingsConfig = {
            modelsCountValue: DEFAULT_MODELS_COUNT_VALUE,
            minSpawnAgentsValue: DEFAULT_MIN_SPAWN_AGENTS_VALUE,
            maxSpawnAgentsValue: DEFAULT_MAX_SPAWN_AGENTS_VALUE,
            workIntervalValue: DEFAULT_WORK_INTERVAL_VALUE,
            statisticIntervalValue: DEFAULT_STATISTIC_INTERVAL_VALUE,
            modelSourceElementsCountValue: DEFAULT_MODEL_SOURCE_ELEMENTS_COUNT_VALUE,
            minQueueCapacity: DEFAULT_MIN_QUEUE_CAPACITY,
            maxQueueCapacity: DEFAULT_MAX_QUEUE_CAPACITY,
            minDelayCapacity: DEFAULT_MIN_DELAY_CAPACITY,
            maxDelayCapacity: DEFAULT_MAX_DELAY_CAPACITY,
            delayValue: DEFAULT_DELAY_VALUE,
            isQualityOfServiceActive: DEFAULT_IS_QUALITY_OF_SERVICE_ACTIVE,
            loadFactorDangerValue: DEFAULT_LOAD_FACTOR_DANGER_VALUE,
            packetLostDangerValue: DEFAULT_PACKET_LOST_DANGER_VALUE,
            pingDangerValue: DEFAULT_PING_DANGER_VALUE,
            jitterDangerValue: DEFAULT_JITTER_DANGER_VALUE,
            isMLBalancingActive: false,
        }
    }

    public getModelsList(): TModelsList {
        return this.modelsList;
    }

    public getControllersList(): TControllersList {
        return this.controllersList;
    }

    public getBalancer(): TBoardBalancer {
        return this.balancer;
    }

    public getModelById(modelId: TModelID): Model {
        const currentModel = this.modelsList.find((model) => model.getID() === modelId);

        if (!currentModel) {
            throw new Error("Cannot get model by id, id is uncorrect");
        }

        return currentModel;
    }

    public getWorkTime(): TBoardTime {
        return this.workTime;
    }

    public getStatisticTime(): TBoardTime {
        return this.statisticTime;
    }

    public getModelsWorkTimer(): TModelsInterval {
        return this.modelsWorkTimer;
    }

    public getSendModelsStatisticTimer(): TModelsInterval {
        return this.sendModelsStatisticTimer;
    }

    public getisModelsCreate(): boolean {
        return this.isModelsCreate;
    }

    public getIsModelStart(): boolean {
        return this.isModelsStart;
    }

    public getIsModelStop(): boolean {
        return this.isModelsStop;
    }

    public getSendingData(): TControllersStatesList {
        return this.sendingData;
    }

    public getSendFunction(): any {
        return this.sendFunction;
    }

    public getSendedBoarData(modelsList: TModelsList) {
        const sendedChartsDataList: TSendedChartsDataList = [];
        const sendedModelsAdditionalInfoList: TSendedModelsAdditionalInfoList = []

        const { workIntervalValue, delayValue } = this.settingsConfig;

        const delayValueToIntervalValueMultiplier = workIntervalValue / delayValue;

        modelsList.forEach((model) => {
            const lastModelStateInfo: IModelStateInfo = model.getModelStateInfo(this.statisticTime);

            const currentLoadFactor: number = ModelStatisticService.getLoadFactor(lastModelStateInfo, delayValueToIntervalValueMultiplier);
            const currentQueueLoad: number = ModelStatisticService.getQueueLoad(lastModelStateInfo);

            const agentsCameInModelCount =
                ModelStatisticService.getAgentsCameInModelCount(lastModelStateInfo);
            const agentsLeftThroughModelCount =
                ModelStatisticService.getAgentsLeftThroughModelCount(lastModelStateInfo);
            const agentsInModelCount =
                ModelStatisticService.getAgentsInModelCount(lastModelStateInfo);
            const agentsLostCount =
                ModelStatisticService.getAgentsLostInModelCount(lastModelStateInfo);

            sendedChartsDataList.push({
                time: String(this.statisticTime / MILLISECONDS_TO_SECONDS_MULTIPLIER),
                loadFactor: String(currentLoadFactor),
                queueLoad: String(currentQueueLoad),
            });

            sendedModelsAdditionalInfoList.push(
                BoardSettingsConfigService.getModelsAdditionalInfo(agentsCameInModelCount, agentsLeftThroughModelCount, agentsInModelCount, agentsLostCount)
            )
        })

        return {
            sendedChartsDataList,
            sendedModelsAdditionalInfoList,
        }
    }

    public getSettingsConfig(): ISettingsConfig {
        return this.settingsConfig;
    }

    public getModelsRatings(): TModelsRatings {
        const modelsRatings: TModelsRatings = [];

        this.modelsList.forEach((model) => {
            const currentQueueCapacity: number = model.getQueueElements()[0].getCapacity();
            const currentDelayCapacity: number = model.getDelayElements()[0].getCapacity();
            const currentGeneralCapacity = currentQueueCapacity + currentDelayCapacity;

            const capacityData: IModelRatingInfo = {
                queue: {
                    currentValue: currentQueueCapacity,
                    maximumValue: this.settingsConfig.maxQueueCapacity,
                    info: ModelRatingInfoList.QUEUE,
                },
                delay: {
                    currentValue: currentDelayCapacity,
                    maximumValue: this.settingsConfig.maxDelayCapacity,
                    info: ModelRatingInfoList.DELAY,
                },
                general: {
                    currentValue: currentGeneralCapacity,
                    maximumValue: this.settingsConfig.maxQueueCapacity + this.settingsConfig.maxDelayCapacity,
                    info: ModelRatingInfoList.GENERAL,
                }

            };

            modelsRatings.push(capacityData);
        });

        return modelsRatings;
    }

    public getDelayCapacitiesList(): number[] {
        return this.modelsList.map((model) => model.getDelayElements()[0].getCapacity());
    }

    public setModelsList(modelsList: TModelsList): void {
        this.modelsList = modelsList;
    }

    public setControllersList(controllersList: TControllersList): void {
        this.controllersList = controllersList;
    }

    public setBalancer(balancer: Balancer): void {
        this.balancer = balancer;
    }

    public setWorkTime(workTime: TBoardTime): void {
        this.workTime = workTime;
    }

    public setStatisticTime(statisticTime: TBoardTime): void {
        this.statisticTime = statisticTime;
    }

    public setModelsWorkTimer(modelsWorkTimer: TModelsInterval): void {
        this.modelsWorkTimer = modelsWorkTimer;
    }

    public setSendModelsStatisticTimer(sendModelsStatisticTimer: TModelsInterval): void {
        this.sendModelsStatisticTimer = sendModelsStatisticTimer;
    }

    public setIsModelsCreate(isModelsCreate: boolean) {
        this.isModelsCreate = isModelsCreate;
    }

    public setIsModelsStart(isModelsStart: boolean) {
        this.isModelsStart = isModelsStart;
    }

    public setIsModelsStop(isModelsStop: boolean) {
        this.isModelsStop = isModelsStop;
    }

    public setSendingData(sendingData: TControllersStatesList) {
        this.sendingData = sendingData;
    }

    public setSendFunction(sendFunction: any) {
        this.sendFunction = sendFunction;
    }

    public updateSettingsConfig(sendedSettingsConfig: ISendableBoardSettingsConfig): void {
        this.settingsConfig.modelsCountValue = sendedSettingsConfig.modelsSettings.fields["modelsCountValue"].value as number;
        this.settingsConfig.minSpawnAgentsValue = sendedSettingsConfig.modelsSettings.fields["minSpawnAgentsValue"].value as number;
        this.settingsConfig.maxSpawnAgentsValue = sendedSettingsConfig.modelsSettings.fields["maxSpawnAgentsValue"].value as number;
        this.settingsConfig.workIntervalValue = sendedSettingsConfig.modelsSettings.fields["workIntervalValue"].value as number;
        this.settingsConfig.statisticIntervalValue = sendedSettingsConfig.modelsSettings.fields["statisticIntervalValue"].value as number;
        this.settingsConfig.modelSourceElementsCountValue = sendedSettingsConfig.modelsSettings.fields["modelSourceElementsCountValue"].value as number;
        this.settingsConfig.minQueueCapacity = sendedSettingsConfig.modelsSettings.fields["minQueueCapacity"].value as number;
        this.settingsConfig.maxQueueCapacity = sendedSettingsConfig.modelsSettings.fields["maxQueueCapacity"].value as number;
        this.settingsConfig.minDelayCapacity = sendedSettingsConfig.modelsSettings.fields["minDelayCapacity"].value as number;
        this.settingsConfig.maxDelayCapacity = sendedSettingsConfig.modelsSettings.fields["maxDelayCapacity"].value as number;
        this.settingsConfig.delayValue = sendedSettingsConfig.modelsSettings.fields["delayValue"].value as number;
        this.settingsConfig.loadFactorDangerValue = sendedSettingsConfig.modelsSettings.fields["loadFactorDangerValue"].value as number;
        this.settingsConfig.packetLostDangerValue = sendedSettingsConfig.qualityOfServiceSettings.fields["packetLostDangerValue"].value as number;
        this.settingsConfig.pingDangerValue = sendedSettingsConfig.qualityOfServiceSettings.fields["pingDangerValue"].value as number;
        this.settingsConfig.jitterDangerValue = sendedSettingsConfig.qualityOfServiceSettings.fields["jitterDangerValue"].value as number;
        this.settingsConfig.isQualityOfServiceActive = sendedSettingsConfig.qualityOfServiceSettings.isActive!;
        this.settingsConfig.isMLBalancingActive = sendedSettingsConfig.mlBalancingSettings.isActive!;
    }

    public addModel(model: Model): void {
        this.modelsList.push(model)
    }

    public addController(controller: Controller): void {
        this.controllersList.push(controller);
    }

    public clearIntervalStatistic(): void {
        this.modelsList.forEach((model) => {
            model.clearIntervalStatistic();
        })
    }

    public clearSendingData(): void {
        this.sendingData = [];
    }

    public async balancerCheck(): Promise<void> {
        if (!this.balancer) {
            throw new Error("Cannot complete statistic interval action, balancer is undefined");
        }

        const { isQualityOfServiceActive, delayValue, workIntervalValue, maxSpawnAgentsValue, loadFactorDangerValue, packetLostDangerValue, pingDangerValue, jitterDangerValue } = this.settingsConfig;

        const delayValueToIntervalValueMultiplier = workIntervalValue / delayValue;

        // Check if it's an ML balancer
        if (this.balancer instanceof MLBalancer) {
            await this.balancer.checkModelsLoadFactors(isQualityOfServiceActive, this.statisticTime, this.sendFunction, delayValueToIntervalValueMultiplier, loadFactorDangerValue, maxSpawnAgentsValue, packetLostDangerValue, pingDangerValue, jitterDangerValue);
        } else {
            this.balancer.checkModelsLoadFactors(isQualityOfServiceActive, this.statisticTime, this.sendFunction, delayValueToIntervalValueMultiplier, loadFactorDangerValue, maxSpawnAgentsValue, packetLostDangerValue, pingDangerValue, jitterDangerValue);
        }
    }

    public modelsIntervalAction(): void {
        this.clearIntervalStatistic();

        this.modelsList.forEach((model) => {
            model.spawnAgents(this.settingsConfig.minSpawnAgentsValue, this.settingsConfig.maxSpawnAgentsValue);
        })

        this.workTime += this.settingsConfig.workIntervalValue;

        console.log(`\n\nWORK TIME: ${this.workTime} ms\n`);
    }

    public statisticIntervalAction(): void {
        this.statisticTime += this.settingsConfig.statisticIntervalValue;

        this.controllersList.forEach((controller) => {
            controller.addNewParametersState(this.statisticTime);
        })

        this.balancerCheck();

        const {sendedChartsDataList, sendedModelsAdditionalInfoList} = this.getSendedBoarData(this.modelsList);

        this.sendFunction(ServerMessageTypes.MODELS_STATES, sendedChartsDataList);
        this.sendFunction(ServerMessageTypes.MODELS_ADDITIONAL_INFO, sendedModelsAdditionalInfoList);
    }

    public create(): void {
        this.modelsList = [];
        this.controllersList = [];

        // Choose balancer type based on settings
        if (this.settingsConfig.isMLBalancingActive) {
            this.balancer = new MLBalancer();
        } else {
            this.balancer = new Balancer();
        }

        const { modelsCountValue, minQueueCapacity, maxQueueCapacity, minDelayCapacity, maxDelayCapacity } = this.settingsConfig

        for (let index = 0; index < modelsCountValue; index++) {
            const newModel = new Model();

            const sourceElements: SourceElement[] = [];
            const networkElements: NetworkElement[] = [];
            const queueElements: QueueElement[] = [];
            const delayElements: DelayElement[] = [];

            const queueElement = new QueueElement();
            const delayElement = new DelayElement();
            const sinkElement = new SinkElement();
            const lostSinkElement = new SinkElement();

            for (let index = 0; index < this.settingsConfig.modelSourceElementsCountValue; index++) {
                const sourceElement = new SourceElement();

                sourceElement.setPreviousElements(getPreviousElementsList());
                sourceElement.setNextElement(queueElement);

                queueElement.setPreviousElements(getPreviousElementsList(sourceElement));

                addElementsInList(sourceElements, sourceElement);
                addElementsInList(networkElements, sourceElement);
            }

            addElementsInList(networkElements, queueElement, delayElement, sinkElement);
            addElementsInList(queueElements, queueElement);
            addElementsInList(delayElements, delayElement);

            delayElement.setPreviousElements(getPreviousElementsList(queueElement));
            sinkElement.setPreviousElements(getPreviousElementsList(delayElement));
            lostSinkElement.setPreviousElements(getPreviousElementsList(queueElement));

            queueElement.setNextElement(delayElement);
            delayElement.setNextElement(sinkElement);

            queueElement.setCapacity(getRandomArbitrary(minQueueCapacity, maxQueueCapacity));
            delayElement.setCapacity(getRandomArbitrary(minDelayCapacity, maxDelayCapacity));

            this.sendFunction(ServerMessageTypes.MESSAGE, `Model ${index + 1} characteristic: queueCapacity = ${queueElement.getCapacity()}, delayCapacity = ${delayElement.getCapacity()}`);

            queueElement.sendListenerInit();
            queueElement.setLostSinkElement(lostSinkElement);
            delayElement.setDelayValue(this.settingsConfig.delayValue);

            newModel.setSourceElements(sourceElements);
            newModel.setNetworkElements(networkElements);
            newModel.setQueueElements(queueElements);
            newModel.setDelayElements(delayElements);
            newModel.setSinkElement(sinkElement);

            this.addModel(newModel);

            const newController = new Controller();

            newController.setServicedModel(newModel);

            this.addController(newController);
            this.balancer.addController(newController);

            console.log("\nCREATE SUCCESS\n");
        }

        this.isModelsStart = false;
    }

    public start(): void {
        if (this.isModelsStart) {
            return;
        }

        this.clearSendingData();

        this.controllersList.forEach((controller) => {
            controller.setParametersStatesList([]);
        })

        this.modelsWorkTimer = setInterval(() => this.modelsIntervalAction(), this.settingsConfig.workIntervalValue);
        this.sendModelsStatisticTimer = setInterval(() => this.statisticIntervalAction(), this.settingsConfig.statisticIntervalValue);

        this.isModelsCreate = true;
        this.isModelsStop = false;
        this.isModelsStart = true;

        console.log("\nSTART SUCCESS\n");
    }

    public stop(): void {
        if (this.isModelsStop) {
            return;
        }

        if (!this.modelsWorkTimer || !this.sendModelsStatisticTimer) {
            throw new Error("Cannot stop models, models has not been started yet");
        }

        clearInterval(this.modelsWorkTimer);
        clearInterval(this.sendModelsStatisticTimer);

        this.modelsList.forEach((model) => {
            model.stop();
        })

        this.workTime = 0;
        this.statisticTime = 0;

        this.isModelsCreate = false;
        this.isModelsStart = false;
        this.isModelsStop = true;

        console.log("\nSTOP SUCCESS\n");

        this.controllersList.forEach((controller) => {
            this.sendingData.push(controller.getParametersStatesList());
        });
    }
}

export default Board;