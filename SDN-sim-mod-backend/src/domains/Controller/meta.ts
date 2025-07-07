import Model from "../Model";

export type TControllerID = string;
export type TServicedModel = Model | null;
export type TControllerParameter = number;

export interface IParametersState {
    time: TControllerParameter,
    CPU: TControllerParameter
    usedDiskSpace: TControllerParameter,
    memoryUsage: TControllerParameter,
    networkTraffic: TControllerParameter,
    packetLost: TControllerParameter,
    ping: TControllerParameter,
    jitter: TControllerParameter,
}
export type TParametersStatesList = IParametersState[];

export const DEFAULT_PARAMETERS_DANGER_VALUE: number = 1;
export const MAX_PARAMETER_LOAD_VALUE: number = 10;