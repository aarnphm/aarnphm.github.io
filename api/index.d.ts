import type { ServerResponse, IncomingMessage } from "http"

export type VercelRequestCookies = { [key: string]: string }
export type VercelRequestQuery = { [key: string]: string | string[] }
export type VercelRequestBody = any

export type VercelRequest = IncomingMessage & {
  query: VercelRequestQuery
  cookies: VercelRequestCookies
  body: VercelRequestBody
}

export type VercelResponse = ServerResponse & {
  send: (body: any) => VercelResponse
  json: (jsonBody: any) => VercelResponse
  status: (statusCode: number) => VercelResponse
  redirect: (statusOrUrl: string | number, url?: string) => VercelResponse
}
