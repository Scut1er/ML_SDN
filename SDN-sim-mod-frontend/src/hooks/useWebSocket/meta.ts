export type TSendMessage = (messageData: any) => void;

export const WS_PROTOCOL: string = "ws://";

export interface IClientMessage {
    commandID: string;
    commandInfo: any;
}