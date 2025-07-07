import { useState } from "react";
import { ISendableBoardSettingsConfig, TBoardSettingsConfigRanges } from "@context/BoardSettingsContext/meta";

const useBoardSettings = () => {
    const [settingsConfigRanges, setSettingsConfigRanges] = useState<TBoardSettingsConfigRanges | null>(null);
    const [settingsConfig, setSettingsConfig] = useState<ISendableBoardSettingsConfig | null>(null);

    const updateBoardSettingsConfig = (newConfig: ISendableBoardSettingsConfig): void => {
        setSettingsConfig(newConfig);
    }

    const updateBoardSettingsConfigRanges = (newConfig: TBoardSettingsConfigRanges): void => {
        setSettingsConfigRanges(newConfig);
    }

    return {
        settingsConfigRanges: settingsConfigRanges,
        setSettingsConfigRanges: setSettingsConfigRanges,
        settingsConfig: settingsConfig,
        setSettingsConfig: setSettingsConfig,
        updateBoardSettingsConfig: updateBoardSettingsConfig,
        updateBoardSettingsConfigRanges: updateBoardSettingsConfigRanges,
    }
}

export default useBoardSettings;
