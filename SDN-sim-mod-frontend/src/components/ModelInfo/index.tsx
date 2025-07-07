import TooltipBlock from "@components/TooltipBlock";
import { ISendedModelAdditionalInfo } from "@hooks/useServerMessageHandler/meta";
import { ModelInfoFieldsNames } from "./meta";
import { ReactElement } from "react";
import "./style.css";

export interface IModelInfoProps {
    info: ISendedModelAdditionalInfo;
}

const ModelInfo = ({ info }: IModelInfoProps): ReactElement => {
    const {
        agentsCameInModelCount,
        agentsLeftThroughModelCount,
        agentsInModelCount,
        agentsLostCount,
    } = info;

    return (
        <div className="model-info-container">
            <div className="came-in-model model-info-field">
                <div className="agents-info">
                    <div className="agents-info-value-title">
                        {ModelInfoFieldsNames.AGENTS_CAME_IN_MODEL_COUNT}
                    </div>
                    <div className="agents-info-value">
                        {agentsCameInModelCount.value}
                    </div>
                </div>
                <div className="info">
                    <TooltipBlock info={agentsCameInModelCount.info} />
                </div>
            </div>
            <div className="in-model model-info-field">
                <div className="agents-info">
                    <div className="agents-info-value-title">
                        {ModelInfoFieldsNames.AGENTS_IN_MODEL_COUNT}
                    </div>
                    <div className="agents-info-value">
                        {agentsInModelCount.value}
                    </div>
                </div>
                <div className="info">
                    <TooltipBlock info={agentsInModelCount.info} />
                </div>
            </div>
            <div className="left-through-model model-info-field">
                <div className="agents-info">
                    <div className="agents-info-value-title">
                        {ModelInfoFieldsNames.AGENTS_LEFT_THROUGH_MODEL_COUNT}
                    </div>
                    <div className="agents-info-value">
                        {agentsLeftThroughModelCount.value}
                    </div>
                </div>
                <div className="info">
                    <TooltipBlock info={agentsLeftThroughModelCount.info} />
                </div>
            </div>
            <div className="lost model-info-field">
                <div className="agents-info">
                    <div className="agents-info-value-title">
                        {ModelInfoFieldsNames.AGENTS_LOST_COUNT}
                    </div>
                    <div className="agents-info-value">
                        {agentsLostCount.value}
                    </div>
                </div>
                <div className="info">
                    <TooltipBlock info={agentsLostCount.info} />
                </div>
            </div>
        </div>
    );
};

export default ModelInfo;
