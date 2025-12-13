// check if dev or prod
export const server = process.env.NODE_ENV === "development" ? "http://localhost:8000" : process.env.NEXT_PUBLIC_SERVER_URL;
