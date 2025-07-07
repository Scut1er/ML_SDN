import { type TSendedChartsDataList } from "@hooks/useServerMessageHandler/meta";
import {
    ChartColors,
    ChartLabelsNames,
    LOAD_FACTOR_AXIS_TITLE,
    QUEUE_LOAD_AXIS_TITLE,
    TIME_AXIS_TITLE,
    type IChartData,
} from "./meta";
import { ChartService } from "@services/ChartService";
import { Chart as ChartJS } from "chart.js/auto";
import { type ReactElement } from "react";
import { Line } from "react-chartjs-2";
import "./style.css";

ChartJS.register();

export const enum ChartDataTypes {
    LOAD_FACTOR = "loadFactor",
    QUEUE_LOAD = "queueLoad",
}

export interface IChartsListProps {
    chartsDataList: TSendedChartsDataList;
    queueCapacity: number;
}

const LineCharts = ({
    chartsDataList,
    queueCapacity,
}: IChartsListProps): ReactElement => {
    const loadFactorChartData: IChartData = ChartService.getChartData(
        ChartLabelsNames.LOAD_FACTOR_FROM_TIME,
        chartsDataList,
        ChartDataTypes.LOAD_FACTOR,
        ChartColors.BLUE
    );

    const queueLoadChartData: IChartData = ChartService.getChartData(
        ChartLabelsNames.QUEUE_LOAD_FROM_TIME,
        chartsDataList,
        ChartDataTypes.QUEUE_LOAD,
        ChartColors.GREEN
    );

    const loadFactorChartOptions = ChartService.getChartOptions(
        false,
        true,
        TIME_AXIS_TITLE,
        true,
        LOAD_FACTOR_AXIS_TITLE,
        true
    );
    const queueLoadChartOptions = ChartService.getChartOptions(
        false,
        true,
        TIME_AXIS_TITLE,
        true,
        QUEUE_LOAD_AXIS_TITLE,
        true,
        queueCapacity
    );

    return (
        <div className="charts-container">
            <Line data={loadFactorChartData} options={loadFactorChartOptions} />
            <Line data={queueLoadChartData} options={queueLoadChartOptions} />
        </div>
    );
};

export default LineCharts;
