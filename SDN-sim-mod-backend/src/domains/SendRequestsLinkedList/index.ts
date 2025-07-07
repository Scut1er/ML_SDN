import { SendRequestsLinkedListNode } from "../SendRequestsLinkedListNode";
import { TBatchedFunction } from "../SendRequestsLinkedListNode/meta";

export class SendRequestsLinkedList {
    private firstNodeInQueue: SendRequestsLinkedListNode | null;
    public length: number;

    constructor() {
        this.firstNodeInQueue = null;
        this.length = 0;
    }

    public addFunction(addingFunction: TBatchedFunction): void {
        const newFunctionInQueue = new SendRequestsLinkedListNode(addingFunction);

        if (!this.firstNodeInQueue) {
            this.firstNodeInQueue = newFunctionInQueue;

            this.length++;

            return;
        }

        let functionState: SendRequestsLinkedListNode = this.firstNodeInQueue;

        while (functionState.nextNode) {
            functionState = functionState.nextNode;
        }

        functionState.nextNode = newFunctionInQueue;

        this.length++;
    }

    public getFirstFunctionInQueue(): TBatchedFunction {
        const completeNode = this.firstNodeInQueue;

        if (!completeNode || !completeNode.currentFunction) {
            throw new Error("Cannot get first function in Queue, first function is undefined");
        }

        const completeFunction = completeNode.currentFunction;

        this.firstNodeInQueue = completeNode.nextNode;

        this.length--;

        return completeFunction;
    }
}