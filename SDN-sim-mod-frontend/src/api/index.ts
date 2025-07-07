const HTTP_API_URL: string = "http://localhost:5500";

export default class API {
    public static async getAccessWsConnectionByUrl(url: string) {
        const response = await fetch(`${HTTP_API_URL}/check-url/${url}`, {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        });

        if (!response.ok) {
            throw new Error(`${response.status}, ${response.statusText}`);
        }

        return await response.json();
    }

    public static async getModelsControllerParametersLists() {
        const response = await fetch(`${HTTP_API_URL}/get-controller-parameters`, {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        });

        if (!response.ok) {
            throw new Error(`${response.status}, ${response.statusText}`);
        }

        return await response.json();
    }
}