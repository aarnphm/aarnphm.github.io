#!/usr/bin/env node
import express from "express"
const app = express()
const PORT = 8001

app.use(express.json())

app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*") // Or specify your frontend's URL instead of '*'
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
  next()
})

app.all("/api", async (req, res) => {
  const apiUrl = "https://curius.app/api/users/3584/searchLinks"
  try {
    const response = await fetch(apiUrl, {
      headers: { "Content-Type": "application/json" },
    })
    res.json(await response.json())
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.listen(PORT, () => console.log(`Server running on port ${PORT}`))
