export type TChartLabels = number[];

export interface IDataset {
    label: string;
    data: number[];
    borderColor?: string;
    pointBackgroundColor?: string;
    pointBorderColor?: string;
    borderWidth?: number;
    pointBorderWidth?: number;
}

export type TDatasets = IDataset[];

export interface IChartData {
    labels: TChartLabels;
    datasets: TDatasets;
}

export const enum ChartLabelsNames {
    LOAD_FACTOR_FROM_TIME = "Load factor from time",
    QUEUE_LOAD_FROM_TIME = "Queue load from time",
}

export const enum ChartColors {
    BLUE = "blue",
    GREEN = "green",
}

export const TIME_AXIS_TITLE: string = "Time per second";
export const LOAD_FACTOR_AXIS_TITLE: string = "Load factor";
export const QUEUE_LOAD_AXIS_TITLE: string = "Queue load";

export interface IChartAxisTitle {
    display: boolean;
    text: string;
}

export interface IChartAxis {
    title: IChartAxisTitle;
}

export interface IChartScales {
    x: IChartAxis;
    y: IChartAxis;
}

export interface IChartOptions {
    animation: boolean;
    scales: IChartScales;
}

