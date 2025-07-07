import ControlToggleSlider from "@components/ControlToggleSlider";
import { type ReactElement } from "react";
import {
    IRangeSettingData,
    ISendableBoardSettingsConfigBlock,
    ISendableSettingsConfigField,
} from "@context/BoardSettingsContext/meta";
import "./style.css";
import BoardSettingsFieldControl from "@components/BoardSettingsFieldControl";

export interface IBoardSettingsBlockProps {
    id: string;
    data: ISendableBoardSettingsConfigBlock;
    getRangeData: (fieldId: string) => IRangeSettingData;
    updateField: any;
    updateActiveStatus: any;
}

const BoardSettingsBlock = ({
    id,
    data,
    getRangeData,
    updateField,
    updateActiveStatus,
}: IBoardSettingsBlockProps): ReactElement => {
    const isNeedActiveCheckbox: boolean = !(data.isActive === undefined);

    const isFieldsActive: boolean = !isNeedActiveCheckbox
        ? true
        : data.isActive!;

    const fieldsKeysList: string[] = Object.keys(data.fields);
    const fieldsValuesList: ISendableSettingsConfigField[] = Object.values(
        data.fields
    );

    const onActiveStatusChange = (value: boolean): void => {
        updateActiveStatus(id, value);
    };

    // const onFieldValueChange = (fieldId: string, value: number): void => {
    //     updateField(id, fieldId, value);
    // };

    return (
        <div className="board-settings-block">
            <div className="header">
                <h3>{data.title}</h3>
                {isNeedActiveCheckbox && (
                    <ControlToggleSlider
                        initialValue={data.isActive!}
                        onChange={onActiveStatusChange}
                    />
                )}
            </div>
            {isFieldsActive &&
                fieldsKeysList.map((key, index) => (
                    <BoardSettingsFieldControl
                        key={key}
                        fieldData={fieldsValuesList[index]}
                        rangeData={getRangeData(key)}
                        onChange={(value: number) => {
                            updateField(id, key, value);
                        }}
                    />
                ))}
        </div>
    );
};

export default BoardSettingsBlock;
