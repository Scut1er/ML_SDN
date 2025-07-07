export const enum DialogButtonsLabels {
    OK = "OK",
    PROCEED = "PROCEED",
    SUBMIT = "SUBMIT",
    DELETE = "DELETE",
    CANCEL = "CANCEL",
};

export interface IDialogOptions {
    title?: string;
    message: string;
    submitLabel: string;
    cancelLabel?: string;
    submitAction: () => void;
    cancelAction?: () => void;
}
export default interface IDialogData {
    id: string;
    title?: string;
    message: string;
    submitLabel: string;
    cancelLabel?: string;
    submitAction: () => void;
    cancelAction?: () => void;
}
