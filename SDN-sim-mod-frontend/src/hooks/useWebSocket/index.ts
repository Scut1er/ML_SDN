import { useState } from "react";
import { type IClientMessage, WS_PROTOCOL } from "./meta";
import { setIsLoading } from "@store/slices/application";
import { useDispatch } from "react-redux";
import { AlertTypes } from "@domains/Alert";

const useWebSocket = (webSocketUrl: string, handler: (data: any) => void, createAlert: any) => {
    const dispatch = useDispatch();

    const [webSocket, setWebSocket] = useState<WebSocket>();

    const sendMessage = (messageData: IClientMessage) => {
        if (!webSocket) {
            console.log("Cannot send message to server, web socket in not connected");

            return;
        }

        webSocket.send(JSON.stringify(messageData));

        dispatch(setIsLoading(true));
    };

    const onOpen = () => {
        console.info("WEBSOCKET CREATED");

        createAlert({
            title: "Websocket status",
            message: "Websocket connection opened",
            type: AlertTypes.SUCCESS,
        })
    };

    const onMessage = (event: MessageEvent) => {
        handler(event.data);

        dispatch(setIsLoading(false));
    };

    const onError = (event: Event) => {
        console.info("ERROR ON WEBSOCKET", event);

        createAlert({
            title: "Websocket status",
            message: "Error on Websocket",
            type: AlertTypes.ERROR,
        })

        dispatch(setIsLoading(false));
    };

    const onClose = () => {
        console.info("WEBSOCKET CLOSED");

        createAlert({
            title: "Websocket status",
            message: "Websocket closed. Reload page to reconnect",
            type: AlertTypes.WARNING,
        })
    };

    const configure = () => {
        const socket = new WebSocket(`${WS_PROTOCOL}${webSocketUrl}`);

        socket.onopen = onOpen;
        socket.onmessage = onMessage;
        socket.onerror = onError;
        socket.onclose = onClose;

        setWebSocket(socket);
    };

    return { configure, sendMessage };
};

export default useWebSocket;
