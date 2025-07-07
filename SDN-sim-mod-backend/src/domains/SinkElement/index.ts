import Agent from "../Agent";
import NetworkElement from "../NetworkElement";
import { startDate } from "../..";
import { DEFAULT_LAST_AGENTS_STATISTIC, ILastAgentsStatistic } from "./meta";
import { ISurroundingNetworkElements } from "../meta";

class SinkElement extends NetworkElement {
    private lastAgentsStatistic: ILastAgentsStatistic;

    constructor() {
        super();
        this.capacity = Infinity;
        this.nextElement = null;
        this.agentsLeftCount = 0;
        this.lastAgentsStatistic = DEFAULT_LAST_AGENTS_STATISTIC;
    }

    private updateStatisticForControllers(newAgent: Agent): void {
        this.lastAgentsStatistic.previousAgentLeftTime = this.lastAgentsStatistic.lastAgentLeftTime;
        
        this.lastAgentsStatistic.lastAgentCameTime = newAgent.getCameTime();
        this.lastAgentsStatistic.lastAgentLeftTime = newAgent.getLeftTime();
    }

    protected sinkAgent(newAgent: Agent): void {
        this.agentsCount = 0;

        const startTime = startDate.getTime();
        const leftTime = (new Date()).getTime();

        newAgent.setLeftTime(leftTime - startTime);
        newAgent.setIsLeftModel(true);

        this.updateStatisticForControllers(newAgent);

        this.removeAgentFromList(newAgent);
    }

    public trigger(initiator: NetworkElement, newAgent: Agent): boolean {
        this.takeAgents(initiator, newAgent);
        this.sinkAgent(newAgent);

        return true;
    }

    public getSurroundingElements(): ISurroundingNetworkElements {
        if (!this.previousElements) {
            throw new Error("Cannot get surrounding elements, previous elements is undefined");
        }

        return {
            previousElements: this.previousElements,
        }
    }

    public getStatisticForController(): ILastAgentsStatistic {
        return this.lastAgentsStatistic;
    }

    public getCurrentState() {
        return { 
            agentsCameCount: this.agentsCameCount,
         }
    }

    public setStatisticForController(lastAgentsStatistic: ILastAgentsStatistic): void {
        this.lastAgentsStatistic = lastAgentsStatistic;
    }
}

export default SinkElement;