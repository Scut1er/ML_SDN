import { Dispatch, SetStateAction } from "react";

export interface IChartsContext {
    isChartsCurrentDotsViewType: boolean;
    chartsDotsCount: number;
    setIsChartsCurrentDotsViewType: Dispatch<SetStateAction<boolean>>;
    setChartsDotsCount: Dispatch<SetStateAction<number>>;
}