import ControlRangeSlider from "@components/ControlRangeSlider";
import {
    type IRangeSettingData,
    type ISendableSettingsConfigField,
} from "@context/BoardSettingsContext/meta";
import { type ReactElement } from "react";
import "./style.css";

export interface IBoardSettingsFieldControlProps {
    fieldData: ISendableSettingsConfigField;
    rangeData: IRangeSettingData;
    onChange: any;
}

const BoardSettingsFieldControl = ({
    fieldData,
    rangeData,
    onChange,
}: IBoardSettingsFieldControlProps): ReactElement => (
    <div className="board-settings-field-control">
        <ControlRangeSlider
            initialValue={fieldData.value}
            label={fieldData.label}
            info={fieldData.info}
            minValue={rangeData.minValue}
            maxValue={rangeData.maxValue}
            step={rangeData.step}
            onChange={onChange}
        />
    </div>
);

export default BoardSettingsFieldControl;
