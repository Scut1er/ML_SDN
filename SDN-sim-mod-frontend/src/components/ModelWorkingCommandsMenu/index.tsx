import * as XLSX from "xlsx";
import API from "@api/index";
import useNotifications from "@hooks/useNotifications";
import ControlButton from "@components/ControlButton";
import BoardWorkContext from "@context/BoardWorkContext";
import BoardSettingsContext from "@context/BoardSettingsContext";
import ModelWorkingCommandButton from "@components/ModelWorkingCommandButton";
import { type IClientMessage } from "@hooks/useWebSocket/meta";
import { type ReactElement, useContext } from "react";
import { setIsLoading } from "@store/slices/application";
import { AlertTypes } from "@domains/Alert";
import { useDispatch } from "react-redux";
import {
    EXCEL_FILE_NAME,
    getWorkbook,
} from "./meta";
import "./style.css";

const DOWNLOAD_ICON_SIZE: number = 15;

const ModelWorkingCommandsMenu = (): ReactElement => {
    const dispatch = useDispatch();

    const { createAlert } = useNotifications();
    const {
        boardWorkCommandsConfig,
        modelsActionsStatesList,
        sendCommandFunction,
        setIsBoardControlPanelOpen,
    } = useContext(BoardWorkContext);
    const { setIsAccessGetResults, settingsConfig, isAccessGetResults } =
        useContext(BoardSettingsContext);

    const closeBoardControlPanel = (): void => {
        setIsBoardControlPanelOpen(false);
    };

    const handleSendCommandToServer = (
        commandKey: string,
        isSendBoardSettingsConfig: boolean,
        isCloseBoardControlPanel: boolean,
        isAccessGetResults: boolean
    ): void => {
        sendCommandFunction({
            commandID: commandKey,
            commandInfo: isSendBoardSettingsConfig ? settingsConfig : "",
        } as IClientMessage);

        setIsAccessGetResults(isAccessGetResults);

        if (!isCloseBoardControlPanel) {
            return;
        }

        closeBoardControlPanel();
    };

    const loadControllersStatesList = async () => {
        try {
            dispatch(setIsLoading(true));

            const data = await API.getModelsControllerParametersLists();

            const workbook = getWorkbook(data);

            XLSX.writeFile(workbook, EXCEL_FILE_NAME, { compression: true });
        } catch (error) {
            createAlert({
                title: "Download status",
                message: "Error on download result of experiment",
                type: AlertTypes.ERROR,
            });
        } finally {
            dispatch(setIsLoading(false));
        }
    };

    const download = (): void => {
        loadControllersStatesList();
    };

    return (
        <div className="model-working-commands menu-container">
            {boardWorkCommandsConfig.map((boardWorkCommandData, index) => (
                <ModelWorkingCommandButton
                    key={boardWorkCommandData.commandKey}
                    actionState={modelsActionsStatesList[index]}
                    commandData={boardWorkCommandData}
                    onClickAction={handleSendCommandToServer}
                />
            ))}
            <ControlButton
                onClick={download}
                title={
                    <svg
                        version="1.1"
                        xmlns="http://www.w3.org/2000/svg"
                        width={DOWNLOAD_ICON_SIZE}
                        height={DOWNLOAD_ICON_SIZE}
                        viewBox="0 0 512 512"
                    >
                        <path d="M368 224l-128 128-128-128h80v-192h96v192zM240 352h-240v128h480v-128h-240zM448 416h-64v-32h64v32z" fill="white"></path>
                    </svg>
                }
                isActive={isAccessGetResults}
            />
        </div>
    );
};

export default ModelWorkingCommandsMenu;
