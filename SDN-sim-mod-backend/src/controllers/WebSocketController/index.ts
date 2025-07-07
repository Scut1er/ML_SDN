import WebSocket from "ws";
import Board from "../../domains/Board";
import BoardSettingsConfigService from "../../services/BoardSettingsConfigService";
import { Client } from "../../domains/Client";
import { randomUUID } from "crypto";
import { boardWorkCommandsConfig, ClientCommandsTypes, COMMAND_INFO_WITHOUS_SETTINGS_CONFIG, IActionConfig, IClientMessage, IServerMessage, ServerInfoMessageTexts, ServerMessageTypes } from "./meta";
import { WEB_CLIENT_PORT } from "../../utils/constants";
import { ISendableBoardSettingsConfig, TBoardSettingsConfigRanges } from "../../domains/Board/meta";

let sendableBoardSettingsConfig: ISendableBoardSettingsConfig = BoardSettingsConfigService.getDefaultBoardSettingsConfig();
let boardSettingsConfigRanges: TBoardSettingsConfigRanges = BoardSettingsConfigService.getBoardSettingsConfigRanges();

export const WebSocketController = (board: Board, startDate: Date) => {
    let clientsList: Client[] = [];

    const MODEL_WORKING_COMMANDS = Object.values(ClientCommandsTypes);

    const sendMessageHostClient = (messageType: string, message: any): void => {
        const hostSocket = clientsList[0].getSocket();

        if (!hostSocket) {
            throw new Error("Cannot send message for host, host is undefined");
        }

        const serverMessage: IServerMessage = {
            messageType: messageType,
            message: message,
        }

        hostSocket.send(JSON.stringify(serverMessage));
    }

    const sendMessageCurrentClient = (messageType: string, message: any, webSocketClient: WebSocket): void => {
        const serverMessage: IServerMessage = {
            messageType: messageType,
            message: message,
        }

        webSocketClient.send(JSON.stringify(serverMessage));
    }

    const sendMessageAllClients = (messageType: string, message: any) => {
        const serverMessage: IServerMessage = {
            messageType: messageType,
            message: message,
        }

        clientsList.forEach((client) => {
            const clientSocket = client.getSocket();

            if (!clientSocket) {
                throw new Error("Cannot send clear charts message, someone client is undefined");
            }

            clientSocket.send(JSON.stringify(serverMessage));
        });
    }

    const sendModelsActionsStates = (webSocketClient: WebSocket): void => {
        const isModelsCreate = board.getisModelsCreate();
        const isModelsStart = board.getIsModelStart();
        const isModelsStop = board.getIsModelStop();

        const modelsActionsStatesMessage: IServerMessage = {
            messageType: ServerMessageTypes.BOARD_ACTIONS_STATES,
            message: [isModelsCreate, isModelsStart, isModelsStop],
        }

        webSocketClient.send(JSON.stringify(modelsActionsStatesMessage));
    }

    const sendModelsRatings = (webSocketClient: WebSocket): void => {
        const modelsRatingsMessage: IServerMessage = {
            messageType: ServerMessageTypes.BOARD_CAPACITIES_LIST,
            message: board.getModelsRatings(),
        }

        webSocketClient.send(JSON.stringify(modelsRatingsMessage));
    }

    const ActionsConfigsList: Record<ClientCommandsTypes, IActionConfig> = {
        [ClientCommandsTypes.CREATE]: {
            updateBoardFunction: () => { board.updateSettingsConfig(sendableBoardSettingsConfig) },
            boardActionFunction: () => { board.create() },
            clientSendActionFunctions: [sendModelsActionsStates],
            allClientsSendActionFunctions: [() => { sendMessageAllClients(ServerMessageTypes.MESSAGE, ServerInfoMessageTexts.CREATE_MODELS) },() => {sendMessageAllClients(ServerMessageTypes.BOARD_CAPACITIES_LIST, board.getModelsRatings())}],
        },
        [ClientCommandsTypes.START]: {
            updateBoardFunction: null,
            boardActionFunction: () => {
                board.start();
                startDate = new Date();
            },
            clientSendActionFunctions: [sendModelsActionsStates],
            allClientsSendActionFunctions: [() => { sendMessageAllClients(ServerMessageTypes.CLEAR_CHARTS, "") },
            () => { sendMessageAllClients(ServerMessageTypes.MESSAGE, ServerInfoMessageTexts.START_MODELS) }
            ],
        },
        [ClientCommandsTypes.STOP]: {
            updateBoardFunction: null,
            boardActionFunction: () => { board.stop() },
            clientSendActionFunctions: [sendModelsActionsStates],
            allClientsSendActionFunctions: [() => { sendMessageAllClients(ServerMessageTypes.MESSAGE, ServerInfoMessageTexts.STOP_MODELS) }],
        },
    }

    const boardAction = (updateBoardFunction: any, boardActionFunction: () => void, clientSendActionFunctions: any[], allClientsSendActionFunctions: any[], webSocketClient: WebSocket): void => {
        if (updateBoardFunction) {
            updateBoardFunction();
        }
        boardActionFunction();

        clientSendActionFunctions.forEach((clientSendActionFunction) => {
            clientSendActionFunction(webSocketClient);
        });

        allClientsSendActionFunctions.forEach((allClientsSendActionFunction) => allClientsSendActionFunction());
    }

    const webSocketClientSetup = (webSocketClient: WebSocket, client: Client): void => {
        sendMessageCurrentClient(ServerMessageTypes.MESSAGE, ServerInfoMessageTexts.CONNECT_MESSAGE, webSocketClient);

        if (board.getModelsList().length) {
            sendModelsRatings(webSocketClient);
        }

        if (clientsList.length) {
            clientsList.push(client);

            console.log("Another client connected to the server");
            return;
        }

        client.setIsHost(true);

        clientsList.push(client);

        sendMessageCurrentClient(ServerMessageTypes.BOARD_WORKING_COMMANDS, boardWorkCommandsConfig, webSocketClient);
        sendMessageCurrentClient(ServerMessageTypes.BOARD_SETTINGS_CONFIG_RANGES, boardSettingsConfigRanges, webSocketClient);
        sendMessageCurrentClient(ServerMessageTypes.BOARD_SETTINGS_CONFIG, sendableBoardSettingsConfig, webSocketClient);

        sendModelsActionsStates(webSocketClient);

        sendMessageCurrentClient(ServerMessageTypes.MESSAGE, ServerInfoMessageTexts.GET_SETUP_DATA, webSocketClient);

        console.log("First client connected to server and get all commands!");
    }

    const handleClientCommand = (webSocketClient: WebSocket, message: IClientMessage) => {
        if (!MODEL_WORKING_COMMANDS.includes(message.commandID as ClientCommandsTypes)) {
            console.log("Unexpected command from client!");
            return;
        };

        if (message.commandInfo !== COMMAND_INFO_WITHOUS_SETTINGS_CONFIG) {
            sendableBoardSettingsConfig = message.commandInfo;
        }

        const actionConfigure: IActionConfig = ActionsConfigsList[message.commandID as ClientCommandsTypes];

        const updateBoardFunction = actionConfigure.updateBoardFunction;
        const boardActionFunction = actionConfigure.boardActionFunction;
        const clientSendActionFunction = actionConfigure.clientSendActionFunctions;
        const allClientsSendActionFunctions = actionConfigure.allClientsSendActionFunctions;

        boardAction(updateBoardFunction, boardActionFunction, clientSendActionFunction, allClientsSendActionFunctions, webSocketClient);
    }

    const hanldeClientLeave = (leavedClientID: string) => {
        if (clientsList[0].getID() === leavedClientID) {
            board.stop();

            sendMessageAllClients(ServerMessageTypes.MESSAGE, ServerInfoMessageTexts.STOP_MODELS);
        }

        clientsList = clientsList.filter((client) =>
            client.getID() !== leavedClientID
        );

        console.log(`Another client closed connection!`);
    }

    const getWebSocketErrorStatus = (event: any): void => {
        console.log("WebSocket error!", event.data);
    }

    const webSocketCreateConnection = (): void => {
        const webSocketServer = new WebSocket.Server({ port: WEB_CLIENT_PORT });

        webSocketServer.on("connection", (webSocketClient) => {
            const newClient = new Client();
            const clientID: string = randomUUID();

            newClient.setID(clientID);
            newClient.setSocket(webSocketClient);

            webSocketClientSetup(webSocketClient, newClient);

            webSocketClient.on('message', (event) => {
                const parsedMessage: IClientMessage = JSON.parse(event.toString("utf-8"));

                handleClientCommand(webSocketClient, parsedMessage);
            });
            webSocketClient.on('close', () => { hanldeClientLeave(clientID) });
            webSocketClient.on("error", getWebSocketErrorStatus);
        });
    }

    return {
        webSocketCreateConnection,
        sendMessageAllClients,
    }
}