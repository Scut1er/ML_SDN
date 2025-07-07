import { Dispatch, SetStateAction, useState } from "react";
import { UserStatuses } from "@components/Application/meta";
import {
    TModelsActionsStatesList,
    TServerMessageType,
    TClientAction,
    ServerMessageTypes,
    TModelsRatings,
    TSendedBoardChartsDataList,
    TSendedModelsAdditionalInfoList,
    TSendedChartsDataList,
} from "./meta";
import { WebsocketMessageParser } from "@services/ModelWebSocketService";
import { AlertTypes } from "@domains/Alert";
import { type TUserStatus } from "@components/Application/meta";
import { type ISendableBoardSettingsConfig, type TBoardSettingsConfigRanges } from "@context/BoardSettingsContext/meta";
import { type TBoardWorkCommandsConfig } from "@context/BoardWorkContext/meta";

const useServerMessageHandler = (
    setUserStatus: Dispatch<SetStateAction<TUserStatus>>,
    setStatLength: Dispatch<SetStateAction<number>>,
    updateBoardSettingsConfig: (newConfig: ISendableBoardSettingsConfig) => void,
    updateBoardSettingsConfigRanges: (newConfig: TBoardSettingsConfigRanges) => void,
    createAlert: any,
) => {
    const [boardWorkCommandsConfig, setBoardWorkCommandsConfig] =
        useState<TBoardWorkCommandsConfig>([]);
    const [modelsActionsStatesList, setModelsActionsStatesList] =
        useState<TModelsActionsStatesList>([false, false]);
    const [sendedBoardChartsDataList, setSendedBoardChartsDataList] =
        useState<TSendedBoardChartsDataList>([]);
    const [modelsAdditionalInfoList, setModelsAdditionalInfoList] = useState<TSendedModelsAdditionalInfoList>([]);
    const [modelsRatings, setModelsRatings] = useState<TModelsRatings>([]);

    const deleteFirstModelsStates = (): void => {
        setSendedBoardChartsDataList((prevList) => {
            let newSendedBoardChartsDataList: TSendedBoardChartsDataList =
                prevList.map((sendedModelChartsDataList) => [
                    ...sendedModelChartsDataList,
                ],
                );

            newSendedBoardChartsDataList.forEach((sendedModelChartsDataList) => {
                sendedModelChartsDataList.shift();
            });

            return newSendedBoardChartsDataList;
        });
    };

    const defaultMessageHandler = (messageData: any) => {
        console.info(messageData);

        createAlert({
            title: "Server info message",
            message: messageData,
            type: AlertTypes.INFO,
        })
    };

    const updateBoardWorkCommandsConfig = (
        messageData: TBoardWorkCommandsConfig
    ): void => {
        setBoardWorkCommandsConfig(messageData);

        setUserStatus(UserStatuses.HOST);
    };

    const updateSendedBoardChartsDataList = (
        sendedChartsDataList: TSendedChartsDataList
    ): void => {
        setSendedBoardChartsDataList((prevList) => {
            let newSendedBoardChartsDataList: TSendedBoardChartsDataList =
                prevList.map((sendedModelChartsDataList) => [...sendedModelChartsDataList]);

            if (!newSendedBoardChartsDataList.length) {
                newSendedBoardChartsDataList =
                    sendedChartsDataList.map((chartsData) => [chartsData]);

                return newSendedBoardChartsDataList;
            }

            sendedChartsDataList.forEach(
                (chartsData, index) => {
                    newSendedBoardChartsDataList[index].push(
                        chartsData
                    );
                }
            );

            return newSendedBoardChartsDataList;
        });

        setStatLength((statLength) => statLength + 1);
    };

    const updateModelsAdditionalInfoList = (sendedModelsAdditionalInfoList: TSendedModelsAdditionalInfoList): void => {
        setModelsAdditionalInfoList(sendedModelsAdditionalInfoList);
    }

    const clearChartsDataLists = (): void => {
        setSendedBoardChartsDataList([]);
        setModelsAdditionalInfoList([]);
        setStatLength(0);
    };

    const updateModelsActionsStates = (
        statesList: TModelsActionsStatesList
    ): void => {
        setModelsActionsStatesList(statesList);
    };

    const updateModelsRatings = (sendedModelsRatings: TModelsRatings): void => {
        setModelsRatings(sendedModelsRatings);
    }

    const ActionsInfoList: Record<TServerMessageType, TClientAction> = {
        [ServerMessageTypes.MESSAGE]: defaultMessageHandler,
        [ServerMessageTypes.BOARD_WORKING_COMMANDS]:
            updateBoardWorkCommandsConfig,
        [ServerMessageTypes.MODELS_STATES]: updateSendedBoardChartsDataList,
        [ServerMessageTypes.CLEAR_CHARTS]: clearChartsDataLists,
        [ServerMessageTypes.BOARD_ACTIONS_STATES]: updateModelsActionsStates,
        [ServerMessageTypes.BOARD_CAPACITIES_LIST]: updateModelsRatings,
        [ServerMessageTypes.BOARD_SETTINGS_CONFIG]: updateBoardSettingsConfig,
        [ServerMessageTypes.BOARD_SETTINGS_CONFIG_RANGES]: updateBoardSettingsConfigRanges,
        [ServerMessageTypes.MODELS_ADDITIONAL_INFO]: updateModelsAdditionalInfoList,
    };

    const handleMessageFromServer = (data: string): void => {
        try {
            const { messageData, messageType } =
                WebsocketMessageParser.parseMessage(data);

            ActionsInfoList[messageType](messageData);
        } catch (error) {
            console.log("Failed to parse message from server", error);
        }
    };

    return {
        boardWorkCommandsConfig: boardWorkCommandsConfig,
        setBoardWorkCommandsConfig: setBoardWorkCommandsConfig,
        modelsActionsStatesList: modelsActionsStatesList,
        setModelsActionsStatesList: setModelsActionsStatesList,
        sendedBoardChartsDataList: sendedBoardChartsDataList,
        setSendedBoardChartsDataList: setSendedBoardChartsDataList,
        modelsAdditionalInfoList: modelsAdditionalInfoList,
        setModelsAdditionalInfoList: setModelsAdditionalInfoList,
        handleMessageFromServer: handleMessageFromServer,
        deleteFirstModelsStates: deleteFirstModelsStates,
        modelsRatings: modelsRatings,
        setModelsRatings: setModelsRatings,
    };
};

export default useServerMessageHandler;
