import { defineConfig } from "drizzle-kit"

export default defineConfig({
  schema: "./worker/schema.ts",
  out: "./migrations",
  dialect: "sqlite",
  driver: "d1-http",
  dbCredentials: {
    databaseId: "a267f69f-5fc0-4fbd-a013-d81c4da2783e",
    accountId: process.env.CLOUDFLARE_ACCOUNT_ID!,
    token: process.env.CLOUDFLARE_API_TOKEN!,
  },
})
