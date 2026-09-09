// Standup the mock Ollama on a fixed port for the real-dsh e2e run.
import { createMockOllama } from './mock-ollama.mjs'

const port = Number(process.argv[2] ?? 21434)
const { server, state } = createMockOllama()
server.listen(port, '127.0.0.1', () => {
  console.log(`mock ollama listening on http://127.0.0.1:${port}`)
})
setInterval(() => {
  const { chatCalls, openAICalls, lastChatBody } = state
  console.log(`[mock] chat=${chatCalls} openai=${openAICalls} last.think=${JSON.stringify(lastChatBody?.think)}`)
}, 5000)
