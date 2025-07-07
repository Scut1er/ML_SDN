import IDialogData, { DialogButtonsLabels, type IDialogOptions } from "@domains/Dialog";
import IAlertData, { type IAlertOptions } from "@domains/Alert";
import IdService from "@services/IdService";
import { useDispatch } from "react-redux";
import {
    addAlert,
    removeAlertById,
    addDialog,
    removeDialogById,
} from "@store/slices/notifications";

const DEFAULT_ALERT_TIMEOUT = 5000; // ms

const useNotifications = () => {
    const dispatch = useDispatch();

    const createDialog = (dialogOptions: IDialogOptions): void => {
        const dialogId: string = IdService.generateUniqueId();

        const onDialogSubmit = (): void => {
            dialogOptions.submitAction();
            dispatch(removeDialogById(dialogId));
        };

        const onDialogCancel = (): void => {
            dialogOptions.cancelAction && dialogOptions.cancelAction();
            dispatch(removeDialogById(dialogId));
        };

        const dialog: IDialogData = {
            id: dialogId,
            ...dialogOptions,
        };

        dialog.submitAction = onDialogSubmit;
        dialog.cancelAction = onDialogCancel;

        if (!dialogOptions.cancelLabel && dialogOptions.cancelAction) {
            dialog.cancelLabel = DialogButtonsLabels.CANCEL;
        }

        dispatch(addDialog(dialog));
    };

    const createAlert = (alertOptions: IAlertOptions): void => {
        const alertId: string = IdService.generateUniqueId();
        const timeout: number =
            !alertOptions.timeout || alertOptions.timeout <= 0
                ? DEFAULT_ALERT_TIMEOUT
                : alertOptions.timeout;

        const timeoutId: NodeJS.Timeout = setTimeout(() => {
            dispatch(removeAlertById(alertId));
        }, timeout);

        const alert: IAlertData = {
            id: alertId,
            timeoutId,
            ...alertOptions,
        };

        dispatch(addAlert(alert));
    };

    return {
        createAlert,
        createDialog,
    };
};

export default useNotifications;
