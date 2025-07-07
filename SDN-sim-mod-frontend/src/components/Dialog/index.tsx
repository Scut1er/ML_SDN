import IDialogData from "@domains/Dialog";
import ControlButton from "@components/ControlButton";
import { type ReactElement } from "react";
import "./style.css";

export interface IAlertDialogProps {
    dialogData: IDialogData;
    isCloseButtonShown?: boolean;
}

const Dialog = ({
    dialogData,
    isCloseButtonShown = false,
}: IAlertDialogProps): ReactElement => {
    return (
        <div className="dialog">
            <div className="dialog-head">
                {dialogData?.title && (
                    <h3 className="dialog-title">{dialogData.title}</h3>
                )}
                {isCloseButtonShown && (
                    <ControlButton
                        onClick={dialogData.cancelAction}
                        title={"Close"}
                        isActive={true}
                    />
                )}
            </div>
            <div className="dialog-body">
                <div className="dialog-content">{dialogData.message}</div>
                <div className="dialog-buttons">
                    <ControlButton
                        onClick={dialogData.submitAction}
                        title={dialogData.submitLabel}
                        isActive={true}
                    />
                    {dialogData?.cancelLabel && (
                        <ControlButton
                            onClick={dialogData.cancelAction}
                            title={dialogData.cancelLabel}
                            isActive={true}
                        />
                    )}
                </div>
            </div>
        </div>
    );
};

export default Dialog;
