import BoardSettingsBlock from "@components/BoardSettingsBlock";
import { type ReactElement, useContext } from "react";
import BoardSettingsContext from "@context/BoardSettingsContext";
import {
    IRangeSettingData,
    ISendableBoardSettingsConfig,
    ISendableBoardSettingsConfigBlock,
} from "@context/BoardSettingsContext/meta";
import "./style.css";

const BoardSettings = (): ReactElement => {
    const { settingsConfig, setSettingsConfig, settingsConfigRanges } =
        useContext(BoardSettingsContext);

    if (!settingsConfig || !settingsConfigRanges) {
        return <></>;
    }

    const settingsConfigKeysList: string[] = Object.keys(settingsConfig);
    const settingsConfigValuesList: ISendableBoardSettingsConfigBlock[] =
        Object.values(settingsConfig);

    const settingsConfigRangesKeysList: string[] =
        Object.keys(settingsConfigRanges);
    const settingsConfigRangesValuesList: IRangeSettingData[] =
        Object.values(settingsConfigRanges);

    const getRangeData = (fieldId: string): IRangeSettingData => {
        for (
            let index = 0;
            index < settingsConfigRangesKeysList.length;
            index++
        ) {
            if (settingsConfigRangesKeysList[index] === fieldId) {
                return settingsConfigRangesValuesList[index];
            }
        }

        throw new Error("Cannot get range data, field id is not found");
    };

    const updateSettingsConfigActiveStatus = (
        blockId: string,
        value: boolean
    ): void => {
        const currentConfig: ISendableBoardSettingsConfig = {
            ...settingsConfig,
        };

        currentConfig[blockId as keyof ISendableBoardSettingsConfig].isActive =
            value;

        setSettingsConfig(currentConfig);
    };

    const updateSettingsConfig = (
        blockId: string,
        fieldId: string,
        value: number
    ): void => {
        const currentConfig: ISendableBoardSettingsConfig = {
            ...settingsConfig,
        };

        currentConfig[blockId as keyof ISendableBoardSettingsConfig].fields[
            fieldId as keyof ISendableBoardSettingsConfigBlock
        ].value = value;

        setSettingsConfig(currentConfig);
    };

    return (
        <div className="board-settings">
            {settingsConfigValuesList.map((blockData, index) => (
                <BoardSettingsBlock
                    key={settingsConfigKeysList[index]}
                    id={settingsConfigKeysList[index]}
                    data={blockData}
                    getRangeData={getRangeData}
                    updateField={updateSettingsConfig}
                    updateActiveStatus={updateSettingsConfigActiveStatus}
                />
            ))}
        </div>
    );
};

export default BoardSettings;
