import { TControllersStatesList } from "../../domains/Board/meta";

export const sendModelsControllerParametersList = (res: any, data: TControllersStatesList): void => {
    res.send(data);
}