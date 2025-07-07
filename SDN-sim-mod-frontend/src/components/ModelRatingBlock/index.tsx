import TooltipBlock from "@components/TooltipBlock";
import { memo, type ReactElement } from "react";
import { type IModelRatingInfo } from "@hooks/useServerMessageHandler/meta";
import "./style.css";

export interface IModelRatingBlockProps {
    ratingInfo: IModelRatingInfo;
}

const ModelRatingBlock = memo(
    ({ ratingInfo }: IModelRatingBlockProps): ReactElement => {
        return (
            <div className="model-rating-block">
                <div className="rating-line queue">
                    <div className="rating-value">
                        {`${ratingInfo.queue.currentValue}/${ratingInfo.queue.maximumValue}`}
                    </div>
                    <div className="info">
                        <TooltipBlock info={ratingInfo.queue.info} flow="bottom"/>
                    </div>
                </div>
                <div className="rating-line delay">
                    <div className="rating-value">
                        {`${ratingInfo.delay.currentValue}/${ratingInfo.delay.maximumValue}`}
                    </div>
                    <div className="info">
                        <TooltipBlock info={ratingInfo.delay.info} flow="bottom"/>
                    </div>
                </div>
                <div className="rating-line total-rating">
                    <div className="rating-value">
                        {`${ratingInfo.general.currentValue}/${ratingInfo.general.maximumValue}`}
                    </div>
                    <div className="info">
                        <TooltipBlock info={ratingInfo.general.info} flow="bottom"/>
                    </div>
                </div>
            </div>
        );
    }
);

export default ModelRatingBlock;
