import { useState } from "react";

export const DEFAULT_IS_CHARTS_CURRENT_DOTS_VIEW_TYPE: boolean = false;
export const DEFAULT_CHARTS_DOTS_COUNT: number = 20;

const useChartsContext = () => {
    const [isChartsCurrentDotsViewType, setIsChartsCurrentDotsViewType] = useState<boolean>(DEFAULT_IS_CHARTS_CURRENT_DOTS_VIEW_TYPE);
    const [chartsDotsCount, setChartsDotsCount] = useState<number>(DEFAULT_CHARTS_DOTS_COUNT);

    return {
        isChartsCurrentDotsViewType: isChartsCurrentDotsViewType,
        chartsDotsCount: chartsDotsCount,
        setIsChartsCurrentDotsViewType: setIsChartsCurrentDotsViewType,
        setChartsDotsCount: setChartsDotsCount,
    }
}

export default useChartsContext;
