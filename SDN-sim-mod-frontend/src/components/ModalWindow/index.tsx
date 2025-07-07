import { type ReactElement, type MouseEvent } from "react";
import "./style.css";

export const enum ModalTypes {
    DEFAULT = "DEFAULT",
    SCREEN_LOCKER = "SCREEN_LOCKER",
}

interface IModalLayoutProps {
    children: ReactElement;
    type?: ModalTypes;
    onCLick?: () => void;
}

const ModalWindow = ({
    type = ModalTypes.DEFAULT,
    children,
    onCLick,
}: IModalLayoutProps): ReactElement | null => {
    const isScreenLocked: boolean = type === ModalTypes.SCREEN_LOCKER;

    const className: string = `modal ${
        isScreenLocked ? "locker-wrapper" : "default-wrapper"
    }`;

    const stopClickEventPropagation = (event: MouseEvent<HTMLDivElement>) => {
        event.stopPropagation();
    };

    const handleClick = (): void => {
        if (isScreenLocked || !onCLick) {
            return;
        }

        onCLick();
    };

    return (
        <div className={className} onClick={handleClick}>
            <div
                className="modal-window-content"
                onClick={stopClickEventPropagation}
            >
                {children}
            </div>
        </div>
    );
};

export default ModalWindow;
