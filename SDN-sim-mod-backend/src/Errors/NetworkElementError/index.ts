import NetworkElement from "../../domains/NetworkElement";

class NetworkElementError extends Error {
    private networkElementName: string;
    private networkElementId: string;

    constructor(message: string) {
        let errorMessage: string;

        const callerElement: any = Error.caller;

        const networkElementName: string =
            callerElement?.constructor?.name ?? "";

        const networkElementId: string = callerElement?.getId ?? "";

        if (!networkElementId || !networkElementName) {
            errorMessage = message;
        } else {
            errorMessage = `[${networkElementName}#${networkElementId}]: ${message}`;
        }

        super(errorMessage);

        this.networkElementName = networkElementName;
        this.networkElementId = networkElementId;
    }

    public getNetworkElementName(): string {
        return this.networkElementName;
    }

    public getNetworkElementId(): string {
        return this.networkElementId;
    }
}

export default NetworkElement;
