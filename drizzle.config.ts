import { defineConfig } from "drizzle-kit"

export default defineConfig({
  schema: "./worker/schema.ts",
  out: "./migrations",
  dialect: "sqlite",
  driver: "d1-http",
  dbCredentials: {
    databaseId: "b29f7130-bb30-447d-a84a-7092c9e98b02",
    accountId: process.env.CLOUDFLARE_ACCOUNT_ID!,
    token: process.env.CLOUDFLARE_API_TOKEN!,
  },
})
