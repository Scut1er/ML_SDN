import { createSlice, PayloadAction, Slice } from "@reduxjs/toolkit";

type Alert = any;
type Dialog = any;

export interface INotificationsSlice {
    dialogs: Alert[];
    alerts: Alert[];
}

const initialState: INotificationsSlice = {
    alerts: [],
    dialogs: [],
};

export const notificationsSlice: Slice<INotificationsSlice> = createSlice({
    name: "notifications",
    initialState,
    reducers: {
        addDialog: (state, action: PayloadAction<Dialog>) => {
            const currentDialogsList: Dialog[] = [...state.dialogs];

            currentDialogsList.push(action.payload);

            state.dialogs = currentDialogsList;
        },
        removeDialogById: (state, action: PayloadAction) => {
            const currentDialogsList: Dialog[] = [...state.dialogs];

            const dialogToDelete: Dialog | undefined = currentDialogsList.find(
                (dialog: Dialog) => dialog.id === action.payload
            );

            if (!dialogToDelete) {
                return;
            }

            currentDialogsList.splice(
                currentDialogsList.indexOf(dialogToDelete),
                1
            );

            state.dialogs = currentDialogsList;
        },
        addAlert: (state, action: PayloadAction<Alert>) => {
            const currentAlertsList: Alert[] = [...state.alerts];

            currentAlertsList.push(action.payload);

            state.alerts = currentAlertsList;
        },
        removeAlertById: (state, action: PayloadAction<string>) => {
            const currentAlertsList: Alert[] = [...state.alerts];

            const alertToDelete: Alert | undefined = currentAlertsList.find(
                (alert: Alert) => alert.id === action.payload
            );

            if (!alertToDelete) {
                return;
            }

            currentAlertsList.splice(
                currentAlertsList.indexOf(alertToDelete),
                1
            );

            state.alerts = currentAlertsList;
        },
    },
});

export const { addDialog, removeDialogById, addAlert, removeAlertById } =
    notificationsSlice.actions;

export default notificationsSlice.reducer;
