import EventEmitter from "events";
import NetworkElement from "./NetworkElement";
import Agent from "./Agent";
import Model from "./Model";
import { TModelStatesInfo, TStatesInfo } from "./Model/meta";

export type TSystem = "system";
export type TNetworkElementInitiator = NetworkElement | TSystem;
export type TNetworkElementId = string;
export type TPreviousNetworkElements = Map<string, NetworkElement> | null;
export type TNextNetworkElement = NetworkElement | null;
export type TNetworkElementCapacity = number;
export type TNetworkElementAgentsCount = number;
export type TNetworkElementAgentsCameCount = number;
export type TNetworkElementAgentsLeftCount = number;
export type TNetworkElementAgentsLostCount = number;
export type TStateValue =
    | TPreviousNetworkElements
    | TNextNetworkElement
    | TNetworkElementCapacity
    | TNetworkElementAgentsCount
    | TNetworkElementAgentsCameCount
    | TNetworkElementAgentsLeftCount;
export type TTakeSignal = EventEmitter | null;

export interface ICurrentState {
    [dataField: string]: number;
}

export type TStates = ICurrentState[];

export interface ISurroundingNetworkElements {
    previousElements?: TPreviousNetworkElements;
    nextElement?: TNextNetworkElement;
}

export type TAgentsList = Agent[];
export type TModelsAgentsList = TAgentsList[];

export type TModelsList = Model[];
export type TBoardTime = number;

export type TModelsStatesInfo = TModelStatesInfo[];

export type TAgentsStatesInfo = TStatesInfo[];