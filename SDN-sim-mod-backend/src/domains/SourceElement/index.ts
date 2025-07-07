import Agent from "../Agent";
import { ISurroundingNetworkElements, TNetworkElementInitiator } from "../meta";
import NetworkElement from "../NetworkElement";

class SourceElement extends NetworkElement {
    private receiptIntensity: number;

    constructor() {
        super();
        this.capacity = Infinity;
        this.previousElements = null;
        this.receiptIntensity = 0;
    }

    public trigger(initiator: TNetworkElementInitiator = "system", newAgent: Agent): void {
        if (!this.nextElement) {
            throw new Error("Triggered trigger() into invalid NetworkElement");
        }

        this.setAgentsCount(this.agentsCount + 1);
        this.setAgentsCameCount(this.agentsCameCount + 1);

        this.receiptIntensity++;

        this.nextElement.trigger(this, newAgent);
    }

    public getReceiptIntensity(): number {
        return this.receiptIntensity;
    }

    public getSurroundingElements(): ISurroundingNetworkElements {
        if (!this.nextElement) {
            throw new Error("Cannot get surrounding elements, next element is undefined");
        }

        return {
            nextElement: this.nextElement,
        }
    }

    public getCurrentState() {
        return {
            agentsLeftCount: this.agentsLeftCount,
            receiptIntensity:this.receiptIntensity,
        }
    }

    public setReceiptIntensity(receiptIntensity: number): void {
        this.receiptIntensity = receiptIntensity;
    }

    public clearReceiptIntensity(): void {
        this.receiptIntensity = 0;
    }
}

export default SourceElement;