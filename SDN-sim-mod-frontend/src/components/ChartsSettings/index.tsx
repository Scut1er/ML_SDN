import ChartsContext from "@context/ChartsContext";
import ControlToggleSlider from "@components/ControlToggleSlider";
import ControlRangeSlider from "@components/ControlRangeSlider";
import { ReactElement, useContext } from "react";
import {
    DEFAULT_CHARTS_DOTS_COUNT,
    DEFAULT_IS_CHARTS_CURRENT_DOTS_VIEW_TYPE,
} from "@hooks/useChartsContext";
import "./style.css";

const MIN_CHARTS_DOTS_COUNT: number = 5;
const MAX_CHARTS_DOTS_COUNT: number = 50;
const DOTS_COUNT_SLIDER_STEP: number = 5;
const DOTS_COUNT_SLIDER_INFO: string =
    "This slider set dots count, which visible on charts in working area. This dots count data saved, other data clear.";

const ChartsSettings = (): ReactElement => {
    const {
        isChartsCurrentDotsViewType,
        setChartsDotsCount,
        setIsChartsCurrentDotsViewType,
    } = useContext(ChartsContext);

    return (
        <div className="charts-settings">
            <div className="header">
                <h3>Chart Settings</h3>
                <ControlToggleSlider
                    initialValue={DEFAULT_IS_CHARTS_CURRENT_DOTS_VIEW_TYPE}
                    onChange={setIsChartsCurrentDotsViewType}
                />
            </div>

            {isChartsCurrentDotsViewType && (
                <ControlRangeSlider
                    initialValue={DEFAULT_CHARTS_DOTS_COUNT}
                    label="Dots count in chart"
                    minValue={MIN_CHARTS_DOTS_COUNT}
                    maxValue={MAX_CHARTS_DOTS_COUNT}
                    onChange={setChartsDotsCount}
                    step={DOTS_COUNT_SLIDER_STEP}
                    info={DOTS_COUNT_SLIDER_INFO}
                />
            )}
        </div>
    );
};

export default ChartsSettings;
