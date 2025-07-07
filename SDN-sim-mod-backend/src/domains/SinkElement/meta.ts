export interface ILastAgentsStatistic {
    previousAgentLeftTime: number;
    lastAgentCameTime: number;
    lastAgentLeftTime: number;
}

export const DEFAULT_LAST_AGENTS_STATISTIC: ILastAgentsStatistic = {
    previousAgentLeftTime: 0,
    lastAgentCameTime: 0,
    lastAgentLeftTime: 0,
}