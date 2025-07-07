import NetworkElement from "../NetworkElement";
import Agent from "../Agent";
import { SendRequestsLinkedList } from "../SendRequestsLinkedList";
import { TLostSinkElement } from "./meta";
import { startDate } from "../..";
import { ISurroundingNetworkElements, TNetworkElementAgentsLostCount } from "../meta";

class QueueElement extends NetworkElement {
    private sendRequestsQueue: SendRequestsLinkedList;
    private lostSinkElement: TLostSinkElement;
    private agentsLostCount: TNetworkElementAgentsLostCount;

    constructor() {
        super();

        this.sendRequestsQueue = new SendRequestsLinkedList();
        this.lostSinkElement = null;
        this.agentsLostCount = 0;
    }

    public sendListenerInit(): void {
        const currentNextElement = this.nextElement;

        if (!currentNextElement) {
            throw new Error("Cannot trigger next element, next elements are undefined");
        }

        if (!currentNextElement.takeSignal) {
            return;
        }
        
        currentNextElement.takeSignal.on("takeAvailable", () => {
            if (!this.sendRequestsQueue.length) {
                return;
            }
            
            this.sendRequestsQueue.getFirstFunctionInQueue()();
        }
        );  
    }

    public trigger(initiator: NetworkElement, newAgent: Agent): void {

        const currentNextElement = this.nextElement;

        if (!this.lostSinkElement) {
            throw new Error("Cannot trigger queue element, lost sink element is undefined");
        }

        this.takeAgents(initiator, newAgent);

        if ((this.agentsCount + 1) > this.capacity) {
            // const initiatorAgentsLeftCount = initiator.getAgentsLeftCount();

            // initiator.setAgentsLeftCount(initiatorAgentsLeftCount + 1);

            const lostTime = (new Date()).getTime() - startDate.getTime();

            newAgent.setLeftTime(lostTime);
            newAgent.setIsLeftModel(true);
            newAgent.setIsLost(true);

            this.lostSinkElement.trigger(this,newAgent);
            this.agentsLostCount += 1;

            // this.removeAgentFromList(newAgent);

            return;
        }

        if (!currentNextElement) {
            throw new Error("Cannot trigger next element, next elements are undefined");
        }

        this.sendRequestsQueue.addFunction(() => {
            // this.removeAgentFromList(newAgent);
            currentNextElement.trigger(this, newAgent)
        });

        const isTakeAvailable: boolean = (currentNextElement.getAgentsCount() < currentNextElement.getCapacity());

        if (!isTakeAvailable) {
            return;
        };

        const takingFunction = this.sendRequestsQueue.getFirstFunctionInQueue();

        takingFunction();
    }

    public getSendRequestsQueue(): SendRequestsLinkedList {
        return this.sendRequestsQueue;
    }

    public getLostSinkElement(): TLostSinkElement {
        return this.lostSinkElement;
    }

    public getAgentsLostCount(): TNetworkElementAgentsLostCount {
        return this.agentsLostCount;
    }

    public getSurroundingElements(): ISurroundingNetworkElements {
        if (!this.previousElements || !this.nextElement) {
            throw new Error("Cannot get surroundings elements, previous or next elements are undefined");
        }

        return {
            previousElements: this.previousElements,
            nextElement: this.nextElement,
        }
    }

    public getCurrentState() {
        return {
            agentsCount: this.agentsCount,
            agentsCameCount: this.agentsCameCount,
            agentsLeftCount: this.agentsLeftCount,
            agentsLostCount: this.agentsLostCount,
        }
    }

    public setSendRequestsQueue(newQueue: SendRequestsLinkedList) {
        this.sendRequestsQueue = newQueue;
    }

    public setLostSinkElement(lostSinkElement: TLostSinkElement) {
        this.lostSinkElement = lostSinkElement;
    }

    public setAgentsLostCount(agentsLostCount: TNetworkElementAgentsLostCount): void {
        this.agentsLostCount = agentsLostCount;
    }

    public stop(): void {
        this.sendRequestsQueue = new SendRequestsLinkedList();
    }
}

export default QueueElement;