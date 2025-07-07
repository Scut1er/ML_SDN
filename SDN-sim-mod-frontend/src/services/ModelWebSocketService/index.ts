import { IServerMessage, ServerMessageTypes } from "@hooks/useServerMessageHandler/meta";

export class WebsocketMessageParser {
    public static parseMessage(data: any) {
        const messageData: IServerMessage = JSON.parse(data);
        const messageType: string = this.getMessageType(messageData);

        return { messageData: messageData.message, messageType}  
    }

    private static getMessageType(
        messageData: IServerMessage
    ): ServerMessageTypes {
        const currentServerMessageType =
            messageData?.messageType as ServerMessageTypes;

        return (
            currentServerMessageType ??
            ServerMessageTypes.MESSAGE
        );
    }
}