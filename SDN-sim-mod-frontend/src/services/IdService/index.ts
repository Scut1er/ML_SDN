const HEX_BASE = 0x10000;
const RATIO = 1;
const RADIX = 16;

export default class IdService {
    public static generateUniqueId(): string {
        return `${this.generateRandomHexSegment()}-${this.generateRandomHexSegment()}-${this.generateRandomHexSegment()}-${this.generateRandomHexSegment()}`;
    }

    private static generateRandomHexSegment(): string {
        return Math.floor((RATIO + Math.random()) * HEX_BASE)
            .toString(RADIX)
            .substring(1);
    }
}
