import { IChartData, TChartLabels } from "@components/LineCharts/meta";
import { ISendedChartsData, TSendedChartsDataList } from "@hooks/useServerMessageHandler/meta";
import { Chart as ChartJS } from "chart.js/auto";

export const MILLISECONDS_TO_SECONDS_MULTIPLIER: number = 1000;

const CHART_LINE_WIDTH: number = 1;
const CHART_DOT_WIDTH: number = 0.1;

const loaderChartDotsList: number[] = [1, 4, 2, 8];

export class ChartService {
    public static getChartLabels(chartsDataList: TSendedChartsDataList): TChartLabels {
        const chartLabels = chartsDataList.map((chartsData) => Number(chartsData.time));

        return chartLabels;
    }

    public static getChartValues(
        chartsDataList: TSendedChartsDataList,
        chartDataType: keyof ISendedChartsData
    ): number[] {
        const chartValues = chartsDataList.map((chartsData) =>
            Number(chartsData[chartDataType])
        );

        return chartValues;
    }

    public static getChartData(
        labelName: string,
        chartsDataList: TSendedChartsDataList,
        chartDataType: keyof ISendedChartsData,
        color: string
    ): IChartData {
        const chartLabels = this.getChartLabels(chartsDataList);
        const chartValues = this.getChartValues(chartsDataList, chartDataType);

        return {
            labels: chartLabels,
            datasets: [
                {
                    label: labelName,
                    data: chartValues,
                    borderColor: color,
                    pointBackgroundColor: color,
                    pointBorderColor: color,
                    borderWidth: CHART_LINE_WIDTH,
                    pointBorderWidth: CHART_DOT_WIDTH,
                },
            ],
        };
    }

    public static getChartOptions(isAnimation: boolean, isXAxisDisplay: boolean, XAxisText: string, isYAxisDisplay: boolean, YAxisText: string, isYAxisStartZero?: boolean, maxYAxisValue?: number): any {
        return {
            animation: isAnimation,
            scales: {
                x: {
                    title: {
                        display: isXAxisDisplay,
                        text: XAxisText,
                    },
                },
                y: {
                    title: {
                        display: isYAxisDisplay,
                        text: YAxisText,
                    },
                    beginAtZero: isYAxisStartZero,
                    max: maxYAxisValue,
                },
            }
        }
    }

    public static getLoaderChartData(ctx: CanvasRenderingContext2D): ChartJS {
        const chartJs = new ChartJS(ctx, {
            type: "line",
            data: {
                labels: loaderChartDotsList,
                datasets: [
                    {
                        label: "Loading",
                        fill: false,
                        borderColor: "rgb(47, 137, 234)",
                        borderWidth: 12,
                        tension: 0.4,
                        data: loaderChartDotsList,
                    },
                ],
            },
            options: {
                responsive: true,
                animations: {
                    tension: {
                        duration: 1000,
                        easing: "easeInOutElastic",
                        from: 0,
                        to: 1,
                        loop: true,
                    },
                },
                scales: {
                    x: { display: false },
                    y: { display: false },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false },
                },
            },
        })

        return chartJs;
    }
}
