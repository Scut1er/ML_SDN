export const enum AlertTypes {
    INFO = "info",
    ERROR = "error",
    SUCCESS = "success",
    WARNING = "warning",
}

export interface IAlertOptions {
    message: string;
    type: AlertTypes;
    title?: string;
    timeout?: number;
}

export default interface IAlertData {
    id: string;
    message: string;
    type: AlertTypes;
    title?: string;
    timeout?: number;
    timeoutId?: NodeJS.Timeout;
}
