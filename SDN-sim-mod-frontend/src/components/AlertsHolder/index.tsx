import Alert from "@components/Alert";
import type IAlertData from "@domains/Alert";
import type IAlertOptions from "@domains/Alert";
import { type TRootState } from "@store/index";
import { type ReactElement } from "react";
import { useSelector } from "react-redux";
import "./style.css";

const AlertsHolder = (): ReactElement => {
    const alerts: IAlertData[] = useSelector(
        (state: TRootState) => state.notifications.alerts
    );

    return (
        <div className="alerts-holder">
            {alerts.map((alertOptions: IAlertOptions) => (
                <Alert key={alertOptions.id} alertOptions={alertOptions} />
            ))}
        </div>
    );
};

export default AlertsHolder;
