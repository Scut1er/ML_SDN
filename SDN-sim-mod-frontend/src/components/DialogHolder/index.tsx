import ModalWindow, { ModalTypes } from "@components/ModalWindow";
import { ReactElement, type ReactNode } from "react";

export interface IConnectDialogProps {
    children: ReactElement;
}

const ConnectDialog = ({children}: IConnectDialogProps): ReactNode => {
    return (
        <ModalWindow type={ModalTypes.SCREEN_LOCKER}>
            {children}
        </ModalWindow>
    );
};

export default ConnectDialog;
