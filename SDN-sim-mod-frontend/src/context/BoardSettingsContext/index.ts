import { createContext } from "react";
import { type IBoardSettingsContext } from "./meta";

const BoardSettingsContext = createContext({} as IBoardSettingsContext);

export default BoardSettingsContext;
