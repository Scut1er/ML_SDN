import { ReactElement } from "react";
import "./style.css";

export interface IDefaultLayoutProps {
    children: ReactElement[];
}

const DefaultLayout = ({ children }: IDefaultLayoutProps): ReactElement => (
    <div className="default-layout layout">
        <div className="main-content-container">{children}</div>
    </div>
);

export default DefaultLayout;
