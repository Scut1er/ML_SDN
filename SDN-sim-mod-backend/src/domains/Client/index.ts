import WebSocket from "ws";
import { TClientID, TClientSocket } from "./meta";

export class Client {
    private socket: TClientSocket;
    private isHost: boolean;
    private ID: TClientID;

    constructor() {
        this.socket = null;
        this.isHost = false;
        this.ID = null;
    }

    public getSocket():TClientSocket {
        return this.socket;
    }

    public getIsHost(): boolean {
        return this.isHost;
    }

    public getID(): TClientID {
        return this.ID;
    }

    public setSocket(webSocket: WebSocket): void {
        this.socket = webSocket;
    }
    
    public setIsHost(isHost: boolean): void {
        this.isHost = isHost;
    }

    public setID(ID: string): void {
        this.ID = ID;
    }
}