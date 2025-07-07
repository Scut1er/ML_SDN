import API from "@api/index";
import useWebSocket from "@hooks/useWebSocket";
import ChartContext from "@context/ChartsContext";
import useBoardSettings from "@hooks/useBoardSettings";
import useChartsContext from "@hooks/useChartsContext";
import ModelsInfoList from "@components/ModelsInfoList";
import BoardWorkContext from "@context/BoardWorkContext";
import BoardControlPanel from "@components/BoardControlPanel";
import BoardSettingsContext from "@context/BoardSettingsContext";
import useServerMessageHandler from "@hooks/useServerMessageHandler";
import WebSocketConnectByUrl from "@components/WebSocketConnectByUrl";
import AlertsHolder from "@components/AlertsHolder";
import DialogHolder from "@components/DialogHolder";
import useNotifications from "@hooks/useNotifications";
import FullScreenLoader from "@components/FullScreenLoader";
import WebSocketContext from "@context/WebSocketContext";
import { LayoutsByUserType, UserStatuses } from "./meta";
import { type TUserStatus } from "./meta";
import { useDispatch, useSelector } from "react-redux";
import { Fragment, ReactElement, useEffect, useState } from "react";
import { type TRootState } from "@store/index";
import { setIsLoading } from "@store/slices/application";
import { AlertTypes } from "@domains/Alert";
import "./style.css";

const Application = (): ReactElement => {
    const dispatch = useDispatch();

    const [userStatus, setUserStatus] = useState<TUserStatus>(
        UserStatuses.USER
    );
    const [webSocketUrl, setWebSocketUrl] = useState<string>("");
    const [isConnected, setIsConnected] = useState<boolean>(false);

    const [isBoardControlPanelOpen, setIsBoardControlPanelOpen] =
        useState<boolean>(true);
    const [isAccessGetResults, setIsAccessGetResults] =
        useState<boolean>(false);

    const [statLength, setStatLength] = useState<number>(0);

    const { createAlert } = useNotifications();

    const {
        isChartsCurrentDotsViewType,
        chartsDotsCount,
        setIsChartsCurrentDotsViewType,
        setChartsDotsCount,
    } = useChartsContext();

    const {
        settingsConfigRanges,
        setSettingsConfigRanges,
        settingsConfig,
        setSettingsConfig,
        updateBoardSettingsConfig,
        updateBoardSettingsConfigRanges,
    } = useBoardSettings();

    const {
        boardWorkCommandsConfig,
        setBoardWorkCommandsConfig,
        modelsActionsStatesList,
        setModelsActionsStatesList,
        sendedBoardChartsDataList,
        setSendedBoardChartsDataList,
        modelsAdditionalInfoList,
        setModelsAdditionalInfoList,
        handleMessageFromServer,
        deleteFirstModelsStates,
        modelsRatings,
        setModelsRatings,
    } = useServerMessageHandler(
        setUserStatus,
        setStatLength,
        updateBoardSettingsConfig,
        updateBoardSettingsConfigRanges,
        createAlert
    );

    const { configure, sendMessage } = useWebSocket(
        webSocketUrl,
        handleMessageFromServer,
        createAlert
    );

    const { isLoading } = useSelector((state: TRootState) => state.application);

    useEffect(() => {
        if (!statLength) {
            return;
        }

        if (!isChartsCurrentDotsViewType) {
            return;
        }

        if (statLength <= chartsDotsCount) {
            return;
        }

        deleteFirstModelsStates();

        setStatLength(statLength - 1);
    }, [sendedBoardChartsDataList]);

    const createConfigure = async () => {
        if (webSocketUrl === "") {
            createAlert({
                message: "Cannot connect to server with empty url",
                type: AlertTypes.ERROR,
            });

            return;
        }

        try {
            dispatch(setIsLoading(true));

            const connectionAccess: boolean =
                await API.getAccessWsConnectionByUrl(webSocketUrl);

            if (!connectionAccess) {
                console.info("FAIL CONNECTION URL");

                createAlert({
                    title: "Connection status",
                    message: "Url is not success",
                    type: AlertTypes.WARNING,
                });

                return;
            }

            configure();

            setIsConnected(true);
        } catch (error) {
            createAlert({
                title: "Connection status",
                message: "Connect to server is not success",
                type: AlertTypes.ERROR,
            });
        } finally {
            dispatch(setIsLoading(false));
        }
    };

    const Render = LayoutsByUserType[userStatus];

    return (
        <WebSocketContext.Provider
            value={{
                isConnected: isConnected,
                userStatus: userStatus,
                webSocketUrl: webSocketUrl,
            }}
        >
            <BoardWorkContext.Provider
                value={{
                    sendedBoardChartsDataList: sendedBoardChartsDataList,
                    setSendedBoardChartsDataList: setSendedBoardChartsDataList,
                    modelsAdditionalInfoList: modelsAdditionalInfoList,
                    setModelsAdditionalInfoList: setModelsAdditionalInfoList,
                    boardWorkCommandsConfig: boardWorkCommandsConfig,
                    setBoardWorkCommandsConfig: setBoardWorkCommandsConfig,
                    modelsActionsStatesList: modelsActionsStatesList,
                    setModelsActionsStatesList: setModelsActionsStatesList,
                    sendCommandFunction: sendMessage,
                    modelsRatings: modelsRatings,
                    isBoardControlPanelOpen: isBoardControlPanelOpen,
                    setIsBoardControlPanelOpen: setIsBoardControlPanelOpen,
                    setModelsRatings: setModelsRatings,
                }}
            >
                <BoardSettingsContext.Provider
                    value={{
                        settingsConfigRanges: settingsConfigRanges,
                        setSettingsConfigRanges: setSettingsConfigRanges,
                        settingsConfig: settingsConfig,
                        setSettingsConfig: setSettingsConfig,
                        isAccessGetResults: isAccessGetResults,
                        setIsAccessGetResults: setIsAccessGetResults,
                    }}
                >
                    <ChartContext.Provider
                        value={{
                            isChartsCurrentDotsViewType:
                                isChartsCurrentDotsViewType,
                            chartsDotsCount: chartsDotsCount,
                            setIsChartsCurrentDotsViewType:
                                setIsChartsCurrentDotsViewType,
                            setChartsDotsCount: setChartsDotsCount,
                        }}
                    >
                        <Render asideComponent={<BoardControlPanel />}>
                            <Fragment>
                                {!isConnected ? (
                                    <DialogHolder>
                                        <WebSocketConnectByUrl
                                            webSocketUrl={webSocketUrl}
                                            setWebSocketUrl={setWebSocketUrl}
                                            isConnected={isConnected}
                                            connectFunction={createConfigure}
                                        />
                                    </DialogHolder>
                                ) : (
                                    <ModelsInfoList />
                                )}
                            </Fragment>
                        </Render>
                        <AlertsHolder />
                        {isLoading && <FullScreenLoader />}
                    </ChartContext.Provider>
                </BoardSettingsContext.Provider>
            </BoardWorkContext.Provider>
        </WebSocketContext.Provider>
    );
};

export default Application;
