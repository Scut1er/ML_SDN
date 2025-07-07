import { randomUUID } from "crypto";
import { TBatchedFunction } from "./meta";

export class SendRequestsLinkedListNode {
    private ID: string;
    public currentFunction: TBatchedFunction | null;
    public nextNode: SendRequestsLinkedListNode | null;

    constructor(currentFunction: TBatchedFunction) {
        this.ID = randomUUID();
        this.currentFunction = currentFunction;
        this.nextNode = null;
    }

    public getID(): string {
        return this.ID;
    }
}