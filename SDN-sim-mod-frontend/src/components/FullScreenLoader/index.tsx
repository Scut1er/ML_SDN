import { useEffect, useRef, type ReactElement } from "react";
import { Chart as ChartJS } from "chart.js/auto";
import { ChartService } from "@services/ChartService";
import "./style.css";

ChartJS.register();

const LOADER_CHART_WIDTH: string = "400px";

const FullScreenLoader = (): ReactElement => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    useEffect(() => {
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");

            if (ctx) {
                const myChart = ChartService.getLoaderChartData(ctx);

                return () => {
                    myChart.destroy();
                };
            }
        }
    }, []);

    return (
        <div className="full-screen-loader">
            <div className="loader" style={{width: LOADER_CHART_WIDTH}}>
                <canvas ref={canvasRef}></canvas>
            </div>
        </div>
    );
};

export default FullScreenLoader;
