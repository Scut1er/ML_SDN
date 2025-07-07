import { TModelsRatings, TSendedBoardChartsDataList, TSendedModelsAdditionalInfoList } from "@hooks/useServerMessageHandler/meta";
import { type Dispatch, type SetStateAction } from "react";

export interface IBoardWorkCommandData {
    commandKey: string;
    isSendSettingsConfig: boolean;
    isCloseBoardControlPanel: boolean;
    isAccessGetResults: boolean;
}

export type TBoardWorkCommandsConfig = IBoardWorkCommandData[];

export type TModelsActionsStatesList = boolean[];
export type TSendCommand = any;

export interface IBoardWorkContext {
    sendedBoardChartsDataList: TSendedBoardChartsDataList;
    setSendedBoardChartsDataList: Dispatch<SetStateAction<TSendedBoardChartsDataList>>;
    modelsAdditionalInfoList: TSendedModelsAdditionalInfoList;
    setModelsAdditionalInfoList: Dispatch<SetStateAction<TSendedModelsAdditionalInfoList>>;
    boardWorkCommandsConfig: TBoardWorkCommandsConfig;
    setBoardWorkCommandsConfig: Dispatch<SetStateAction<TBoardWorkCommandsConfig>>;
    modelsActionsStatesList: TModelsActionsStatesList;
    setModelsActionsStatesList: Dispatch<SetStateAction<TModelsActionsStatesList>>;
    modelsRatings: TModelsRatings;
    setModelsRatings: Dispatch<SetStateAction<TModelsRatings>>;
    isBoardControlPanelOpen: boolean;
    setIsBoardControlPanelOpen: Dispatch<SetStateAction<boolean>>;
    sendCommandFunction: TSendCommand;
};


