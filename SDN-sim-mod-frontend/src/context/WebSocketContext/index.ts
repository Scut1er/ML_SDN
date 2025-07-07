import { createContext } from "react";
import { IWebSocketContext } from "./meta";

const WebSocketContext = createContext({} as IWebSocketContext);

export default WebSocketContext;