import { GoogleGenAI } from "@google/genai";
import Anthropic from "@anthropic-ai/sdk";
import Groq from "groq-sdk";
import fs from "node:fs/promises";
import path from "node:path";
import { exec } from "node:child_process";
import { promisify } from "node:util";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

const execAsync = promisify(exec);
const ROOT = path.resolve(process.cwd(), "workspace");
const MAX_STEPS = 40;

const SYSTEM = `You are a practical autonomous coding agent working inside a project workspace.
Turn the user's request into working code, not merely an explanation.
Inspect the workspace before changing it. Create/edit files, run commands, and verify your work.
After implementation, audit for missing pieces, broken imports, obvious runtime/build issues, and incomplete requirements.
Run relevant checks (usually install/build/test) and fix failures. Repeat until the requested work is complete.
Do not claim completion while known build/test errors remain. Keep changes focused.
The workspace root is the only filesystem area you should operate on. Prefer relative paths.
`;

type ToolCall = { id: string; name: string; input: Record<string, unknown> };
type ToolResult = { id: string; result: string };

const tools = [
  { name: "list_files", description: "List files/directories in a workspace directory.", input_schema: { type: "object", properties: { directory: { type: "string" } }, required: [] } },
  { name: "read_file", description: "Read a UTF-8 text file from the workspace.", input_schema: { type: "object", properties: { file_path: { type: "string" } }, required: ["file_path"] } },
  { name: "write_file", description: "Create or completely replace a UTF-8 text file in the workspace.", input_schema: { type: "object", properties: { file_path: { type: "string" }, content: { type: "string" } }, required: ["file_path", "content"] } },
  { name: "edit_file", description: "Replace one exact text fragment in a file. Fails unless the old text occurs exactly once.", input_schema: { type: "object", properties: { file_path: { type: "string" }, old_text: { type: "string" }, new_text: { type: "string" } }, required: ["file_path", "old_text", "new_text"] } },
  { name: "search_files", description: "Search text recursively in workspace text files.", input_schema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } },
  { name: "run_command", description: "Run a shell command from the workspace. Use for installs, builds, tests, linting and debugging.", input_schema: { type: "object", properties: { command: { type: "string" } }, required: ["command"] } }
];

function safePath(p: string) {
  const resolved = path.resolve(ROOT, p || ".");
  if (resolved !== ROOT && !resolved.startsWith(ROOT + path.sep)) throw new Error("Path escapes workspace");
  return resolved;
}

async function listFiles(directory = ".") {
  const dir = safePath(directory);
  const entries = await fs.readdir(dir, { withFileTypes: true });
  return entries.filter(e => !["node_modules", ".git", "dist", ".next"].includes(e.name))
    .map(e => `${e.isDirectory() ? "[dir] " : "      "}${path.relative(ROOT, path.join(dir, e.name)) || "."}`).join("\n") || "(empty)";
}

async function readFile(filePath: string) { return fs.readFile(safePath(filePath), "utf8"); }

async function writeFile(filePath: string, content: string) {
  const target = safePath(filePath);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, content, "utf8");
  return `Wrote ${path.relative(ROOT, target)} (${content.length} characters)`;
}

async function editFile(filePath: string, oldText: string, newText: string) {
  const target = safePath(filePath);
  const current = await fs.readFile(target, "utf8");
  const count = current.split(oldText).length - 1;
  if (count !== 1) throw new Error(`Expected old_text exactly once, found ${count} times`);
  await fs.writeFile(target, current.replace(oldText, newText), "utf8");
  return `Edited ${path.relative(ROOT, target)}`;
}

async function searchFiles(query: string) {
  const results: string[] = [];
  async function walk(dir: string): Promise<void> {
    for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
      if (["node_modules", ".git", "dist", ".next"].includes(entry.name)) continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) await walk(full);
      else {
        try {
          const text = await fs.readFile(full, "utf8");
          if (text.includes(query)) {
            const lines = text.split(/\r?\n/).map((line, i) => line.includes(query) ? `${i + 1}: ${line.slice(0, 300)}` : "").filter(Boolean).slice(0, 8);
            results.push(`${path.relative(ROOT, full)}\n${lines.join("\n")}`);
          }
        } catch { /* ignore binary/unreadable files */ }
      }
      if (results.length >= 50) return;
    }
  }
  await walk(ROOT);
  return results.join("\n\n") || "No matches found.";
}

async function runCommand(command: string) {
  const { stdout, stderr } = await execAsync(command, { cwd: ROOT, timeout: 120_000, maxBuffer: 2_000_000, windowsHide: true });
  return [`STDOUT:\n${stdout}`, stderr ? `STDERR:\n${stderr}` : ""].filter(Boolean).join("\n").slice(0, 12000) || "Command completed with no output.";
}

async function execute(call: ToolCall): Promise<string> {
  switch (call.name) {
    case "list_files": return listFiles(String(call.input.directory ?? "."));
    case "read_file": return readFile(String(call.input.file_path));
    case "write_file": return writeFile(String(call.input.file_path), String(call.input.content));
    case "edit_file": return editFile(String(call.input.file_path), String(call.input.old_text), String(call.input.new_text));
    case "search_files": return searchFiles(String(call.input.query));
    case "run_command": return runCommand(String(call.input.command));
    default: throw new Error(`Unknown tool: ${call.name}`);
  }
}

class GeminiProvider {
  private ai: any;
  constructor(key: string) { this.ai = new GoogleGenAI({ apiKey: key }); }

  async run(userPrompt: string) {
    const history: any[] = [{ role: "user", parts: [{ text: userPrompt }] }];
    for (let step = 1; step <= MAX_STEPS; step++) {
      const response: any = await this.ai.models.generateContent({
        model: process.env.GEMINI_MODEL || "gemini-3.6-flash",
        contents: history,
        config: { systemInstruction: SYSTEM, tools: [{ functionDeclarations: tools.map(t => ({ name: t.name, description: t.description, parametersJsonSchema: t.input_schema })) }] }
      });
      const parts = response.candidates?.[0]?.content?.parts ?? [];
      history.push({ role: "model", parts });
      for (const part of parts) if (part.text) console.log(`\n${part.text}`);
      const calls = parts.filter((p: any) => p.functionCall).map((p: any) => p.functionCall);
      if (!calls.length) return response.text || "Done.";
      const functionResponses = [];
      for (const call of calls) {
        console.log(`\n→ ${call.name}`);
        try {
          const result = await execute({ id: call.id || call.name, name: call.name, input: call.args || {} });
          console.log(`✓ ${result.slice(0, 300)}`);
          functionResponses.push({ functionResponse: { name: call.name, response: { result } } });
        } catch (e) {
          const result = `ERROR: ${e instanceof Error ? e.message : String(e)}`;
          console.log(`✗ ${result}`);
          functionResponses.push({ functionResponse: { name: call.name, response: { result } } });
        }
      }
      history.push({ role: "user", parts: functionResponses });
    }
    throw new Error(`Agent stopped after ${MAX_STEPS} steps. Review the workspace and continue if needed.`);
  }
}

class AnthropicProvider {
  private client: Anthropic;
  constructor(key: string) { this.client = new Anthropic({ apiKey: key }); }

  async run(userPrompt: string) {
    const messages: any[] = [{ role: "user", content: userPrompt }];
    for (let step = 1; step <= MAX_STEPS; step++) {
      const response: any = await this.client.messages.create({
        model: process.env.ANTHROPIC_MODEL || "claude-sonnet-4-20250514",
        max_tokens: 12000,
        system: SYSTEM,
        tools: tools as any,
        messages
      });
      const toolResults: ToolResult[] = [];
      for (const block of response.content) {
        if (block.type === "text") console.log(`\n${block.text}`);
        if (block.type === "tool_use") {
          console.log(`\n→ ${block.name}`);
          try {
            const result = await execute({ id: block.id, name: block.name, input: block.input || {} });
            console.log(`✓ ${result.slice(0, 300)}`);
            toolResults.push({ id: block.id, result });
          } catch (e) {
            const result = `ERROR: ${e instanceof Error ? e.message : String(e)}`;
            console.log(`✗ ${result}`);
            toolResults.push({ id: block.id, result });
          }
        }
      }
      messages.push({ role: "assistant", content: response.content });
      if (!toolResults.length) return response.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("\n");
      messages.push({ role: "user", content: toolResults.map(r => ({ type: "tool_result", tool_use_id: r.id, content: r.result })) });
    }
    throw new Error(`Agent stopped after ${MAX_STEPS} steps. Review the workspace and continue if needed.`);
  }
}

const openAiTools = tools.map(t => ({
  type: "function" as const,
  function: { name: t.name, description: t.description, parameters: t.input_schema }
}));

class GroqProvider {
  private client: Groq;
  constructor(key: string) { this.client = new Groq({ apiKey: key }); }

  async run(userPrompt: string) {
    const messages: any[] = [
      { role: "system", content: SYSTEM },
      { role: "user", content: userPrompt }
    ];
    for (let step = 1; step <= MAX_STEPS; step++) {
      const response = await this.client.chat.completions.create({
        model: process.env.GROQ_MODEL || "openai/gpt-oss-120b",
        messages,
        tools: openAiTools,
      });
      const message = response.choices[0].message;
      if (message.content) console.log(`\n${message.content}`);
      messages.push(message);

      const calls = message.tool_calls;
      if (!calls || !calls.length) return message.content || "Done.";

      for (const call of calls) {
        console.log(`\n→ ${call.function.name}`);
        let result: string;
        try {
          const args = call.function.arguments ? JSON.parse(call.function.arguments) : {};
          result = await execute({ id: call.id, name: call.function.name, input: args });
          console.log(`✓ ${result.slice(0, 300)}`);
        } catch (e) {
          result = `ERROR: ${e instanceof Error ? e.message : String(e)}`;
          console.log(`✗ ${result}`);
        }
        messages.push({ role: "tool", tool_call_id: call.id, content: result });
      }
    }
    throw new Error(`Agent stopped after ${MAX_STEPS} steps. Review the workspace and continue if needed.`);
  }
}

async function ask(question: string) {
  const rl = readline.createInterface({ input, output });
  const answer = (await rl.question(question)).trim();
  rl.close();
  return answer;
}

async function main() {
  await fs.mkdir(ROOT, { recursive: true });
  console.log("\n=== Mini Coding Agent ===\n");
  console.log("1. Gemini (requires billing enabled on your Google account)");
  console.log("2. Anthropic (paid)");
  console.log("3. Groq (free, no billing required — recommended for testing)");
  const choice = await ask("Choose AI provider [1/2/3]: ");
  if (!["1", "2", "3"].includes(choice)) throw new Error("Choose 1, 2, or 3.");
  const envName = choice === "1" ? "GEMINI_API_KEY" : choice === "2" ? "ANTHROPIC_API_KEY" : "GROQ_API_KEY";
  const providerLabel = choice === "1" ? "Gemini" : choice === "2" ? "Anthropic" : "Groq";
  let key = process.env[envName];
  if (!key) key = await ask(`${providerLabel} API key: `);
  if (!key) throw new Error("An API key is required.");
  const request = await ask("\nWhat do you want to build?\n> ");
  if (!request) throw new Error("Describe the work you want done.");
  console.log(`\nWorkspace: ${ROOT}`);
  console.log("Starting agent...\n");
  const provider = choice === "1" ? new GeminiProvider(key) : choice === "2" ? new AnthropicProvider(key) : new GroqProvider(key);
  await provider.run(request);
  console.log("\n✓ Agent finished. Check the workspace for the completed project.\n");
}

main().catch(err => { console.error(`\n✗ ${err instanceof Error ? err.message : String(err)}\n`); process.exit(1); });
