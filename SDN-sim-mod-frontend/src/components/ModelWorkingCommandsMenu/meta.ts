import * as XLSX from "xlsx";
export type TControllerParameter = number;

export interface IParametersState {
    time: TControllerParameter;
    CPU: TControllerParameter;
    usedDiskSpace: TControllerParameter;
    memoryUsage: TControllerParameter;
    networkTraffic: TControllerParameter;
    packetLost: TControllerParameter;
    ping: TControllerParameter;
    jitter: TControllerParameter;
}

export type TParametersStatesList = IParametersState[];

export type TControllersStatesList = TParametersStatesList[];

export type TRowsList = any[];

export const EXCEL_FILE_NAME = "ModelsParametersStatistic.xlsx";

export const ExcelFileTitles = [
    "Time, per seconds",
    "CPU",
    "Used Disk Space",
    "Memory Usage",
    "Network Traffic",
    "Packet Lost",
    "Ping",
    "Jitter",
];

export const DOWNLOAD_BUTTON_TEXT: string = "download";

export const getWorkbook = (rowsList: TRowsList[]) => {
    const workbook = XLSX.utils.book_new();

    rowsList.forEach((rows, index) => {
        const worksheet = XLSX.utils.json_to_sheet(rows);

        XLSX.utils.sheet_add_aoa(worksheet, [ExcelFileTitles]);

        XLSX.utils.book_append_sheet(workbook, worksheet, `Model ${index + 1}`);
    });

    return workbook;
};
