import ModelWorkingCommandsMenu from "../ModelWorkingCommandsMenu";
import { ReactElement } from "react";
import "./style.css";
import BoardSettings from "../BoardSettings";
import ChartsSettings from "../ChartsSettings";

const BoardControlPanel = (): ReactElement => {
    return (
        <div className="board-control-panel">
            <ModelWorkingCommandsMenu />
            <BoardSettings />
            <ChartsSettings />
        </div>
    );
};

export default BoardControlPanel;
