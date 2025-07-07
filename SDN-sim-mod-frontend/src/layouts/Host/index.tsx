import AsideHeader from "@components/AsideHeader";
import BoardWorkContext from "@context/BoardWorkContext";
import { type ReactElement, useContext } from "react";
import "./style.css";

export interface IHostLayoutProps {
    asideComponent: ReactElement;
    children: ReactElement[];
}

const HostLayout = ({
    asideComponent,
    children,
}: IHostLayoutProps): ReactElement => {
    const { isBoardControlPanelOpen, setIsBoardControlPanelOpen } =
        useContext(BoardWorkContext);

    const asideClassNames: string = !isBoardControlPanelOpen
        ? "default"
        : "in-focus";
    const mainClassNames: string = !isBoardControlPanelOpen
        ? "default"
        : "out-of-focus";

    const closeAside = () => {
        setIsBoardControlPanelOpen(false);
    };

    const toggleAside = () => {
        setIsBoardControlPanelOpen(!isBoardControlPanelOpen);
    };

    return (
        <div className="host-layout layout">
            <aside className={asideClassNames}>
                <AsideHeader onToggle={toggleAside} />
                <div className="aside-content-container">{asideComponent}</div>
            </aside>
            <main className={mainClassNames} onClick={closeAside}>
                <div className="main-content-container">{children}</div>
            </main>
        </div>
    );
};

export default HostLayout;
