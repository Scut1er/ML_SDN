import Application from "@components/Application";
import store from "./store";
import ReactDOM, {type Root } from "react-dom/client";
import { Provider } from "react-redux";

const RootHTMLElement: HTMLElement | null = document.getElementById("root");

if (!RootHTMLElement) {
    throw new Error("Cannot acquire div#root to start the Application");
}

const ReactRoot: Root = ReactDOM.createRoot(RootHTMLElement);

ReactRoot.render(
    <Provider store={store}>
        <Application />
    </Provider>
);
