import { type ReactElement } from "react";
import { IAlertOptions } from "@domains/Alert";
import "./style.css";

export interface IAlertProps {
    alertOptions: IAlertOptions;
}

const Alert = ({ alertOptions }: IAlertProps): ReactElement => {
    return (
        <div className={`alert ${alertOptions.type}`}>
            <div className="title">{alertOptions.title}</div>
            {alertOptions.message}
        </div>
    );
};

export default Alert;
