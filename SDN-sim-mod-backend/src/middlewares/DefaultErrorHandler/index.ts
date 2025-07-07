import { type Request, type Response, type NextFunction } from "express";
import { CustomError } from "../../Errors/CustomError";

const DefaultErrorHandler = (
    error: CustomError,
    request: Request,
    response: Response,
    next: NextFunction
) => {
    const { statusCode, message } = error;

    response.status(statusCode).json({
        status: "error",
        statusCode: statusCode,
        message: statusCode === 500 ? "Unexpected server error" : message,
    });

    next();
};

export default DefaultErrorHandler;
