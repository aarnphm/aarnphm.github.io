import { defineConfig } from 'drizzle-kit'

export default defineConfig({
  schema: './worker/schema/flashcards.ts',
  out: './migrations/flashcards',
  dialect: 'sqlite',
  driver: 'd1-http',
  dbCredentials: {
    databaseId: '4fa538cf-e186-417b-922b-dfd88eb92628',
    accountId: process.env.CLOUDFLARE_ACCOUNT_ID!,
    token: process.env.CLOUDFLARE_API_TOKEN!,
  },
})
