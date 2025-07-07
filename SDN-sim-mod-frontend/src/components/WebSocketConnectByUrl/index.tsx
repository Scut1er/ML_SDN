import {
    type Dispatch,
    type ReactElement,
    type SetStateAction,
    type ChangeEvent,
    useState,
} from "react";
import "./style.css";
import ControlButton from "@components/ControlButton";
import TooltipBlock from "@components/TooltipBlock";

export const WEB_SOCKET_CONNECT_INPUT_NAME: string = "URL";
export const WEB_SOCKET_CONNECT_INPUT_INFO: string = "Type current backend URL";

export const DEFAULT_URL: string = "localhost:3001";

export const enum InputNameClasses {
    ACTIVE = "active",
    INACTIVE = "inactive",
}

export const BUTTON_TEXT: string = "connect";

export interface IWebSocketConnectByUrl {
    webSocketUrl: string;
    setWebSocketUrl: Dispatch<SetStateAction<string>>;
    isConnected: boolean;
    connectFunction: () => void;
}

const WebSocketConnectByUrl = ({
    webSocketUrl,
    setWebSocketUrl,
    isConnected,
    connectFunction,
}: IWebSocketConnectByUrl): ReactElement => {
    const [query, setQuery] = useState<string>(webSocketUrl);
    const [inputNameClass, setInputNameClass] = useState<string>(
        InputNameClasses.INACTIVE
    );

    const handleUrlChange = (event: ChangeEvent<HTMLInputElement>): void => {
        setQuery(event.target.value);
        setWebSocketUrl(event.target.value);
    };

    const handleInputFocus = () => {
        setInputNameClass(InputNameClasses.ACTIVE);
    };

    const handleInputBlur = () => {
        setInputNameClass(InputNameClasses.INACTIVE);
    };

    const connect = () => {
        if (isConnected) {
            return;
        }

        connectFunction();
    };

    const useDefaultInputValue = () => {
        setQuery(DEFAULT_URL);
        setWebSocketUrl(DEFAULT_URL);
    };

    return (
        <div className="web-socket-model-interact main-container">
            <div className="web-socket-url-form">
                <div className={`url-input-name ${inputNameClass}`}>
                    {WEB_SOCKET_CONNECT_INPUT_NAME}
                </div>
                <input
                    className="url-input-field"
                    type="text"
                    id="url"
                    value={query}
                    onChange={handleUrlChange}
                    onFocus={handleInputFocus}
                    onBlur={handleInputBlur}
                    placeholder={"URL"}
                    required
                />
                <div
                    data-flow={"top"}
                    data-tooltip={"Use default value"}
                    className="default-input-value"
                    onClick={useDefaultInputValue}
                >
                </div>
                <div className="tooltip-container">
                    <TooltipBlock info={WEB_SOCKET_CONNECT_INPUT_INFO} />
                </div>
            </div>
            <ControlButton
                onClick={connect}
                title={BUTTON_TEXT}
                isActive={!isConnected}
            />
            {/* <button
                className={`common${buttonActiveClass}`}
                onClick={handleClick}
            >
                {BUTTON_TEXT}
            </button> */}
        </div>
    );
};

export default WebSocketConnectByUrl;
