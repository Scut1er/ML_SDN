import express, { type Express } from "express";
import DefaultErrorHandler from "./middlewares/DefaultErrorHandler";
import CORS from "./middlewares/CORS";
import Board from "./domains/Board";
import { WebSocketController } from "./controllers/WebSocketController";
import { PORT } from "./utils/constants";
import { checkUrl } from "./controllers/CheckUrlController";
import { sendModelsControllerParametersList } from "./controllers/ParametersController";

const app: Express = express();
// const port = process.env.PORT || 5500;

app.use(DefaultErrorHandler);
app.use(CORS);

const board = new Board();

export const startDate = new Date();

const { webSocketCreateConnection, sendMessageAllClients } = WebSocketController(board, startDate);

board.setSendFunction(sendMessageAllClients);

webSocketCreateConnection();

app.get(`/check-url/:url`, (req, res) => {
    const { url } = req.params;

    checkUrl(url, res)
});

app.get(`/get-controller-parameters`, (req, res) => {
    sendModelsControllerParametersList(res, board.getSendingData());
});

app.listen(PORT, () => {
    console.log(`[server]: Server is running at http://localhost:${PORT}`);
});
