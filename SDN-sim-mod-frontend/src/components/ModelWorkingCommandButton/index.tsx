import ControlButton from "@components/ControlButton";
import { type ReactElement } from "react";
import {
    type IBoardWorkCommandData,
    type TSendCommand,
} from "@context/BoardWorkContext/meta";
import "./style.css";

export interface IModelWorkingCommandButtonProps {
    actionState: boolean;
    commandData: IBoardWorkCommandData;
    onClickAction: TSendCommand;
}

export const enum WorkingCommandButtonClasses {
    ACTIVE = "active",
    INACTIVE = "inactive",
}

const ModelWorkingCommandButton = ({
    actionState,
    commandData,
    onClickAction,
}: IModelWorkingCommandButtonProps): ReactElement => {
    const handleClick = (): void => {
        if (actionState || !onClickAction) {
            return;
        }

        onClickAction(
            commandData.commandKey,
            commandData.isSendSettingsConfig,
            commandData.isCloseBoardControlPanel,
            commandData.isAccessGetResults
        );

        if (!commandData.isCloseBoardControlPanel) {
            return;
        }
    };

    return (
        <ControlButton
            onClick={handleClick}
            title={commandData.commandKey}
            isActive={!actionState}
        />
    );
};

export default ModelWorkingCommandButton;
