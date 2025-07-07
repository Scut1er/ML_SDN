const WS_URL = "localhost:3001";

export const checkUrl = (url: string, res: any) => {
    return res.send(WS_URL === url);
}