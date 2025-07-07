import ControlToggleSlider from "@components/ControlToggleSlider";
import { type ReactElement } from "react";
import "./style.css";

export interface IMLBalancingToggleProps {
    isActive: boolean;
    onChange: (value: boolean) => void;
    disabled?: boolean;
}

const MLBalancingToggle = ({
    isActive,
    onChange,
    disabled = false,
}: IMLBalancingToggleProps): ReactElement => {
    const handleChange = (value: boolean) => {
        if (!disabled) {
            onChange(value);
        }
    };

    return (
        <div className={`ml-balancing-toggle ${disabled ? 'disabled' : ''}`}>
            <div className="ml-toggle-header">
                <h3>ML Balancing</h3>
                <ControlToggleSlider
                    initialValue={isActive}
                    onChange={handleChange}
                />
            </div>
            <div className="ml-toggle-description">
                <p>Use neural network for load balancing decisions</p>
                {disabled && (
                    <p className="warning">ML Service not available</p>
                )}
            </div>
        </div>
    );
};

export default MLBalancingToggle; 