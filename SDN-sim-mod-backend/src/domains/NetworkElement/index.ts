import Agent from "../Agent";
import {
    type ICurrentState,
    type ISurroundingNetworkElements,
    type TNetworkElementId,
    type TNetworkElementAgentsCameCount,
    type TNetworkElementAgentsCount,
    type TNetworkElementAgentsLeftCount,
    type TNetworkElementCapacity,
    type TNextNetworkElement,
    type TPreviousNetworkElements,
    type TNetworkElementInitiator,
    type TTakeSignal,
    type TAgentsList,
} from "../meta";
import { randomUUID } from "crypto";

abstract class NetworkElement {
    protected id: TNetworkElementId;
    protected agentsList: TAgentsList;
    protected previousElements: TPreviousNetworkElements;
    protected nextElement: TNextNetworkElement;
    protected capacity: TNetworkElementCapacity;
    protected agentsCount: TNetworkElementAgentsCount;
    protected agentsCameCount: TNetworkElementAgentsCameCount;
    protected agentsLeftCount: TNetworkElementAgentsLeftCount;
    public takeSignal: TTakeSignal;

    constructor() {
        this.id = randomUUID();
        this.agentsList = [];
        this.previousElements = null;
        this.nextElement = null;
        this.capacity = 0;
        this.agentsCount = 0;
        this.agentsCameCount = 0;
        this.agentsLeftCount = 0;
        this.takeSignal = null;
    }

    protected getPreviousElementById(id: TNetworkElementId): NetworkElement {
        if (!id) {
            throw new Error("Cannot get element by ID, ID is undefined");
        }

        if (!this.previousElements) {
            throw new Error("Cannot get element by ID, Previous elements is undefined");
        }

        const targetElement: NetworkElement | undefined = this.previousElements.get(id);

        if (!targetElement) {
            throw new Error(`Cannot get element by ID, the Element #${id} does not exist`);
        }

        return targetElement;
    }

    protected takeAgents(sourceNetworkElement: NetworkElement, newAgent: Agent): void {
        // if ((sourceNetworkElement.agentsCount < 1) || ((this.agentsCount + 1) > this.capacity)) {
        //     // return;
        //     throw new Error("Cannot taking agent, NetworkElement is filled");
        // }

        sourceNetworkElement.setAgentsCount(sourceNetworkElement.agentsCount - 1);
        sourceNetworkElement.setAgentsLeftCount(sourceNetworkElement.agentsLeftCount + 1);
        sourceNetworkElement.removeAgentFromList(newAgent);

        this.agentsList.push(newAgent);

        this.setAgentsCameCount(this.agentsCameCount + 1);
        this.setAgentsCount(this.agentsCount + 1);
    }

    public abstract trigger(initiator: TNetworkElementInitiator, newAgent: Agent): void;

    public abstract getSurroundingElements(): ISurroundingNetworkElements;

    public abstract getCurrentState(): ICurrentState;

    public getId(): TNetworkElementId {
        return this.id;
    }

    public getAgentsList(): TAgentsList {
        return this.agentsList;
    }

    public getCapacity(): TNetworkElementCapacity {
        if (!this.capacity) {
            throw new Error(
                "Cannot get capacity, capacity is undefined"
            );
        }

        return this.capacity;
    }

    public getAgentsCount(): TNetworkElementAgentsCount {
        return this.agentsCount;
    }

    public getAgentsCameCount(): TNetworkElementAgentsCameCount {
        return this.agentsCameCount;
    }

    public getAgentsLeftCount(): TNetworkElementAgentsLeftCount {
        return this.agentsLeftCount;
    }

    public getPreviousElements(): TPreviousNetworkElements {
        if (!this.previousElements) {
            throw new Error(
                "Cannot get previous elements, previous elements is undefined"
            );
        }

        return this.previousElements;
    }

    public getNextElement(): TNextNetworkElement {
        if (!this.nextElement) {
            throw new Error(
                "Cannot get next element, next element is undefined"
            );
        }

        return this.nextElement;
    }

    public setAgentsList(agentsList: TAgentsList): void {
        this.agentsList = agentsList;
    }

    public setCapacity(capacity: TNetworkElementCapacity): void {
        this.capacity = capacity;
    }

    public setAgentsCount(agentsCount: TNetworkElementAgentsCount): void {
        this.agentsCount = agentsCount;
    }

    public setAgentsCameCount(agentsCameCount: TNetworkElementAgentsCameCount): void {
        this.agentsCameCount = agentsCameCount;
    }

    public setAgentsLeftCount(agentsLeftCount: TNetworkElementAgentsLeftCount): void {
        this.agentsLeftCount = agentsLeftCount;
    }

    public setPreviousElements(
        previousElements: TPreviousNetworkElements
    ): void {
        this.previousElements = previousElements;
    }

    public setNextElement(nextElement: TNextNetworkElement): void {
        this.nextElement = nextElement;
    }

    public removeAgentFromList(removedAgent: Agent): void {
        this.agentsList = this.agentsList.filter((agent) => agent !== removedAgent);
    }
}

export default NetworkElement;
