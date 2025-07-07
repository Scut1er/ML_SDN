import { TAgentId, TAgentTime } from "./meta";

class Agent {
    private id: TAgentId;
    private cameTime: TAgentTime;
    private leftTime: TAgentTime;
    private isLeftModel: boolean; 
    private isLost: boolean;

    constructor() {
        this.id = 0;
        this.cameTime = 0;
        this.leftTime = 0;
        this.isLeftModel = false;
        this.isLost = false;
    }

    public getId(): TAgentId {
        return this.id;
    }

    public getCameTime(): TAgentTime {
        return this.cameTime;
    }

    public getLeftTime(): TAgentTime {
        return this.leftTime;
    }

    public getIsLeftModel(): boolean {
        return this.isLeftModel;
    }

    public getIsLost(): boolean {
        return this.isLost;
    }

    public getCurrentState() {
        return {
            id: this.id,
            cameTime: this.cameTime,
            leftTime: this.leftTime,
        }
    }

    public setId(id: TAgentId): void {
        this.id = id;
    }

    public setCameTime(cameTime: TAgentTime): void {
        this.cameTime = cameTime;
    }

    public setLeftTime(leftTime: TAgentTime): void {
        this.leftTime = leftTime;
    }

    public setIsLeftModel(isLeftModel: boolean): void {
        this.isLeftModel = isLeftModel;
    }

    public setIsLost(isLost: boolean): void {
        this.isLost = isLost;
    }
}

export default Agent;