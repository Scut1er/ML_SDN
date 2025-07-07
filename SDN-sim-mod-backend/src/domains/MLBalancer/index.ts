import Controller from "../Controller";
import { randomUUID } from "crypto";
import { TBoardTime } from "../meta";
import { TBalancerID } from "./meta";
import { TControllersList } from "../Board/meta";
import { ServerMessageTypes } from "../../controllers/WebSocketController/meta";

interface MLPredictionRequest {
    parametersSettings: {
        criticalLoadFactor: number;
        criticalPacketLost: number;
        criticalPing: number;
        criticalJitter: number;
    };
    modelsParameters: Array<{
        id: string;
        loadFactor: number;
        packetLost: number;
        ping: number;
        jitter: number;
        CPU: number;
        usedDiskSpace: number;
        memoryUsage: number;
        networkTraffic: number;
    }>;
}

interface MLPredictionResponse {
    isNeedIntervene: boolean;
    sendingModelId?: string;
    acceptingModelId?: string;
    confidence: number;
    method: "ML" | "Traditional";
}

class MLBalancer {
    private ID: TBalancerID;
    private controllersList: TControllersList;
    private mlServiceUrl: string;
    private isMLServiceAvailable: boolean;

    constructor(mlServiceUrl: string = "http://localhost:5001") {
        this.ID = randomUUID();
        this.controllersList = [];
        this.mlServiceUrl = mlServiceUrl;
        this.isMLServiceAvailable = false;
        this.checkMLServiceAvailability();
    }

    public getID(): TBalancerID {
        return this.ID;
    }

    public getControllersList(): TControllersList {
        return this.controllersList;
    }

    public setControllersList(controllerList: TControllersList): void {
        this.controllersList = controllerList;
    }

    public addController(controller: Controller): void {
        this.controllersList.push(controller);
    }

    private async checkMLServiceAvailability(): Promise<void> {
        try {
            const response = await fetch(`${this.mlServiceUrl}/health`);
            if (response.ok) {
                const data = await response.json();
                this.isMLServiceAvailable = data.status === "healthy";
                console.log(`ML Service status: ${this.isMLServiceAvailable ? "Available" : "Unavailable"}`);
            } else {
                this.isMLServiceAvailable = false;
            }
        } catch (error) {
            console.log("ML Service not available, falling back to traditional balancing");
            this.isMLServiceAvailable = false;
        }
    }

    private prepareMLRequest(
        workTime: TBoardTime,
        delayValueToIntervalValueMultiplier: number,
        loadFactorDangerValue: number,
        maxSpawnAgentsValue: number,
        packetLostDangerValue: number,
        pingDangerValue: number,
        jitterDangerValue: number
    ): MLPredictionRequest {
        const parametersSettings = {
            criticalLoadFactor: loadFactorDangerValue,
            criticalPacketLost: packetLostDangerValue,
            criticalPing: pingDangerValue,
            criticalJitter: jitterDangerValue
        };

        const modelsParameters = this.controllersList.map((controller, index) => {
            const servicedModel = controller.getServicedModel();
            if (!servicedModel) {
                throw new Error(`Cannot prepare ML request, controller ${index} has no serviced model`);
            }

            const currentLoadFactor = servicedModel.getLoadFactor(workTime, delayValueToIntervalValueMultiplier);
            const currentParametersState = controller.getParametersState(workTime);

            return {
                id: servicedModel.getID(),
                loadFactor: currentLoadFactor,
                packetLost: currentParametersState.packetLost,
                ping: currentParametersState.ping,
                jitter: currentParametersState.jitter,
                CPU: currentParametersState.CPU,
                usedDiskSpace: currentParametersState.usedDiskSpace,
                memoryUsage: currentParametersState.memoryUsage,
                networkTraffic: currentParametersState.networkTraffic
            };
        });

        return {
            parametersSettings,
            modelsParameters
        };
    }

    private async callMLService(requestData: MLPredictionRequest): Promise<MLPredictionResponse | null> {
        try {
            const response = await fetch(`${this.mlServiceUrl}/predict`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                console.error("ML Service returned error:", response.status);
                return null;
            }

            const data = await response.json();
            return data as MLPredictionResponse;
        } catch (error) {
            console.error("Error calling ML Service:", error);
            return null;
        }
    }

    private findControllerByModelId(modelId: string): Controller | null {
        return this.controllersList.find(controller => {
            const servicedModel = controller.getServicedModel();
            return servicedModel?.getID() === modelId;
        }) || null;
    }

    private executeMLDecision(
        prediction: MLPredictionResponse,
        sendFunction: any
    ): void {
        if (!prediction.isNeedIntervene || !prediction.sendingModelId || !prediction.acceptingModelId) {
            return;
        }

        const sendingController = this.findControllerByModelId(prediction.sendingModelId);
        const acceptingController = this.findControllerByModelId(prediction.acceptingModelId);

        if (!sendingController || !acceptingController) {
            console.error("Cannot find controllers for ML decision");
            return;
        }

        const sendingModel = sendingController.getServicedModel();
        const acceptingModel = acceptingController.getServicedModel();

        if (!sendingModel || !acceptingModel) {
            console.error("Cannot find models for ML decision");
            return;
        }

        const sendingModelIndex = this.controllersList.indexOf(sendingController);
        const acceptingModelIndex = this.controllersList.indexOf(acceptingController);

        sendingController.movedServicedModelSinkElement(
            acceptingModel,
            sendFunction,
            sendingModelIndex,
            acceptingModelIndex
        );

        const method = prediction.method || "ML";
        const confidence = (prediction.confidence * 100).toFixed(1);
        
        sendFunction(
            ServerMessageTypes.MESSAGE,
            `[${method}] Source element moved from Model ${sendingModelIndex + 1} to Model ${acceptingModelIndex + 1} (confidence: ${confidence}%)`
        );
    }

    // Fallback to traditional balancing
    private traditionalBalancing(
        isQualityOfServiceActive: boolean,
        workTime: TBoardTime,
        sendFunction: any,
        delayValueToIntervalValueMultiplier: number,
        loadFactorDangerValue: number,
        maxSpawnAgentsValue: number,
        packetLostDangerValue: number,
        pingDangerValue: number,
        jitterDangerValue: number
    ): void {
        const isNeedModelsLoadsAnalysis = this.getNeedModelsLoadsAnalysis(
            isQualityOfServiceActive,
            workTime,
            delayValueToIntervalValueMultiplier,
            loadFactorDangerValue,
            packetLostDangerValue,
            pingDangerValue,
            jitterDangerValue
        );

        if (!isNeedModelsLoadsAnalysis) {
            return;
        }

        const parametersLoadAmountsList: number[] = [];
        this.controllersList.forEach((controller) => {
            parametersLoadAmountsList.push(
                controller.getParametersAmount(workTime, maxSpawnAgentsValue, pingDangerValue, jitterDangerValue)
            );
        });

        const mostLoadedControllerIndex = this.getMaxElementIndex(parametersLoadAmountsList);
        const leastLoadedControllerIndex = this.getMinElementIndex(parametersLoadAmountsList);

        const mostLoadedController = this.controllersList[mostLoadedControllerIndex]!;
        const leastLoadedController = this.controllersList[leastLoadedControllerIndex]!;
        const recipientModel = leastLoadedController.getServicedModel();

        if (!recipientModel) {
            throw new Error("Cannot defined recipient model for move source element, serviced model is undefined");
        }

        mostLoadedController.movedServicedModelSinkElement(
            recipientModel,
            sendFunction,
            mostLoadedControllerIndex,
            leastLoadedControllerIndex
        );

        sendFunction(
            ServerMessageTypes.MESSAGE,
            `[Traditional] Source element moved from Model ${mostLoadedControllerIndex + 1} to Model ${leastLoadedControllerIndex + 1}`
        );
    }

    private getNeedModelsLoadsAnalysis(
        isQualityOfServiceActive: boolean,
        workTime: TBoardTime,
        delayValueToIntervalValueMultiplier: number,
        loadFactorDangerValue: number,
        packetLostDangerValue: number,
        pingDangerValue: number,
        jitterDangerValue: number
    ): boolean {
        if (!isQualityOfServiceActive) {
            for (let index = 0; index < this.controllersList.length; index++) {
                const currentController = this.controllersList[index];
                const servicedModel = currentController.getServicedModel();

                if (!servicedModel) {
                    throw new Error("Cannot check models load factors, some controller has no serviced model");
                }

                const currentLoadFactor = servicedModel.getLoadFactor(workTime, delayValueToIntervalValueMultiplier);

                if (currentLoadFactor < loadFactorDangerValue) {
                    continue;
                }

                return true;
            }
            return false;
        }

        for (let index = 0; index < this.controllersList.length; index++) {
            const currentController = this.controllersList[index];
            const currentParametersState = currentController.getParametersState(workTime);

            if (
                currentParametersState.packetLost < packetLostDangerValue &&
                currentParametersState.ping < pingDangerValue &&
                currentParametersState.jitter < jitterDangerValue
            ) {
                continue;
            }

            return true;
        }

        return false;
    }

    private getMaxElementIndex(array: number[]): number {
        let maxIndex = 0;
        for (let i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private getMinElementIndex(array: number[]): number {
        let minIndex = 0;
        for (let i = 1; i < array.length; i++) {
            if (array[i] < array[minIndex]) {
                minIndex = i;
            }
        }
        return minIndex;
    }

    public async checkModelsLoadFactors(
        isQualityOfServiceActive: boolean,
        workTime: TBoardTime,
        sendFunction: any,
        delayValueToIntervalValueMultiplier: number,
        loadFactorDangerValue: number,
        maxSpawnAgentsValue: number,
        packetLostDangerValue: number,
        pingDangerValue: number,
        jitterDangerValue: number
    ): Promise<void> {
        // Check if ML service is available
        if (!this.isMLServiceAvailable) {
            console.log("ML Service not available, using traditional balancing");
            this.traditionalBalancing(
                isQualityOfServiceActive,
                workTime,
                sendFunction,
                delayValueToIntervalValueMultiplier,
                loadFactorDangerValue,
                maxSpawnAgentsValue,
                packetLostDangerValue,
                pingDangerValue,
                jitterDangerValue
            );
            return;
        }

        try {
            // Prepare ML request
            const mlRequest = this.prepareMLRequest(
                workTime,
                delayValueToIntervalValueMultiplier,
                loadFactorDangerValue,
                maxSpawnAgentsValue,
                packetLostDangerValue,
                pingDangerValue,
                jitterDangerValue
            );

            // Call ML service
            const mlPrediction = await this.callMLService(mlRequest);

            if (!mlPrediction) {
                console.log("ML Service call failed, using traditional balancing");
                this.traditionalBalancing(
                    isQualityOfServiceActive,
                    workTime,
                    sendFunction,
                    delayValueToIntervalValueMultiplier,
                    loadFactorDangerValue,
                    maxSpawnAgentsValue,
                    packetLostDangerValue,
                    pingDangerValue,
                    jitterDangerValue
                );
                return;
            }

            // Execute ML decision
            this.executeMLDecision(mlPrediction, sendFunction);

        } catch (error) {
            console.error("Error during ML balancing:", error);
            // Fallback to traditional balancing
            this.traditionalBalancing(
                isQualityOfServiceActive,
                workTime,
                sendFunction,
                delayValueToIntervalValueMultiplier,
                loadFactorDangerValue,
                maxSpawnAgentsValue,
                packetLostDangerValue,
                pingDangerValue,
                jitterDangerValue
            );
        }
    }
}

export default MLBalancer; 