// import { configDotenv } from "dotenv";
import { Request, Response, NextFunction } from "express";

const ALLOWED_METHODS: string = "GET";

const CORS = (request: Request, response: Response, next: NextFunction) => {
    const method: string = request.method;
    const origin: string | undefined = request.headers["origin"];
    const requestHeaders: string | undefined =
        request.headers["access-control-request-headers"];

    response.header("Access-Control-Allow-Credentials", "true");

    if (origin) {
        response.header("Access-Control-Allow-Origin", origin);
    }

    if (method === "OPTIONS") {
        response.header("Access-Control-Allow-Methods", ALLOWED_METHODS);

        if (requestHeaders) {
            response.header("Access-Control-Allow-Headers", requestHeaders);
        }

        return response.status(200).send();
    }

    next();
};

export default CORS;
