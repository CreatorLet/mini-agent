import { GoogleGenAI } from "@google/genai";
import Anthropic from "@anthropic-ai/sdk";
import Groq from "groq-sdk";
import fs from "node:fs/promises";
import path from "node:path";
import { exec } from "node:child_process";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

const ROOT = path.resolve(process.cwd(), "workspace");
const MAX_STEPS = 50;
const TOOL_RETRY_LIMIT = 2;
const COMMAND_TIMEOUT = 120_000;
const MAX_COMMAND_OUTPUT = 12_000;

const SKIP_DIRS = new Set([
  "node_modules",
  ".git",
  "dist",
  ".next",
  "venv",
  "__pycache__",
]);

const SYSTEM = `You are a practical autonomous coding agent working inside a project workspace.

Your job is to turn the user's request into working code, not merely an explanation.

Workflow:
1. Inspect the workspace before changing anything.
2. Understand the existing project and choose an appropriate implementation.
3. Create or edit files as needed.
4. Install dependencies only when required.
5. Run relevant checks such as builds, tests, linting, or startup checks.
6. Read failures carefully and fix them.
7. Re-run checks after fixes.
8. Audit the final project for missing requirements, broken imports, obvious runtime issues, and unfinished work.
9. Do not claim completion while known build/test/runtime errors remain.

Environment:
- This agent is running on a Windows computer.
- The workspace root is the only filesystem area you may modify.
- Prefer Windows-compatible commands and Node/npm commands.
- Do not use Linux-only commands when a Windows/Node equivalent is available.
- Use relative workspace paths in tool calls.
- Do not access, modify, or delete anything outside the workspace.

Tool-call discipline:
- Always provide valid JSON arguments matching the tool schema.
- For write_file, include the complete file content.
- For edit_file, old_text must match exactly once.
- After making changes, verify them with the appropriate command(s).
`;

type ToolCall = {
  id: string;
  name: string;
  input: Record<string, unknown>;
};

type ToolResult = {
  id: string;
  result: string;
};

type CommandResult = {
  stdout: string;
  stderr: string;
  exitCode: number;
};

const tools = [
  {
    name: "list_files",
    description: "List files/directories in a workspace directory.",
    input_schema: {
      type: "object",
      properties: { directory: { type: "string" } },
      required: [],
      additionalProperties: false,
    },
  },
  {
    name: "read_file",
    description: "Read a UTF-8 text file from the workspace.",
    input_schema: {
      type: "object",
      properties: { file_path: { type: "string" } },
      required: ["file_path"],
      additionalProperties: false,
    },
  },
  {
    name: "write_file",
    description:
      "Create or completely replace a UTF-8 text file in the workspace. Parent directories are created automatically.",
    input_schema: {
      type: "object",
      properties: {
        file_path: { type: "string" },
        content: { type: "string" },
      },
      required: ["file_path", "content"],
      additionalProperties: false,
    },
  },
  {
    name: "edit_file",
    description:
      "Replace one exact text fragment in a file. Fails unless the old text occurs exactly once.",
    input_schema: {
      type: "object",
      properties: {
        file_path: { type: "string" },
        old_text: { type: "string" },
        new_text: { type: "string" },
      },
      required: ["file_path", "old_text", "new_text"],
      additionalProperties: false,
    },
  },
  {
    name: "delete_file",
    description:
      "Delete a file or an empty directory inside the workspace. Use only when it is part of the requested implementation or cleanup.",
    input_schema: {
      type: "object",
      properties: { file_path: { type: "string" } },
      required: ["file_path"],
      additionalProperties: false,
    },
  },
  {
    name: "search_files",
    description: "Search text recursively in workspace text files.",
    input_schema: {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"],
      additionalProperties: false,
    },
  },
  {
    name: "run_command",
    description:
      "Run a shell command from the workspace. Use for installs, builds, tests, linting, startup checks, and debugging.",
    input_schema: {
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
      additionalProperties: false,
    },
  },
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function safePath(inputPath: string) {
  const resolved = path.resolve(ROOT, inputPath || ".");
  if (resolved !== ROOT && !resolved.startsWith(ROOT + path.sep)) {
    throw new Error("Path escapes workspace");
  }
  return resolved;
}

function relativePath(target: string) {
  return path.relative(ROOT, target) || ".";
}

async function listFiles(directory = ".") {
  const dir = safePath(directory);
  const entries = await fs.readdir(dir, { withFileTypes: true });

  return (
    entries
      .filter((entry) => !SKIP_DIRS.has(entry.name))
      .map(
        (entry) =>
          `${entry.isDirectory() ? "[dir] " : "      "}${relativePath(path.join(dir, entry.name))}`,
      )
      .join("\n") || "(empty)"
  );
}

async function readFile(filePath: string) {
  const target = safePath(filePath);
  return fs.readFile(target, "utf8");
}

async function writeFile(filePath: string, content: string) {
  const target = safePath(filePath);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, content, "utf8");
  return `Wrote ${relativePath(target)} (${content.length} characters)`;
}

async function editFile(filePath: string, oldText: string, newText: string) {
  const target = safePath(filePath);
  const current = await fs.readFile(target, "utf8");
  const count = oldText === "" ? 0 : current.split(oldText).length - 1;

  if (count !== 1) {
    throw new Error(`Expected old_text exactly once, found ${count} times`);
  }

  await fs.writeFile(target, current.replace(oldText, newText), "utf8");
  return `Edited ${relativePath(target)}`;
}

async function deleteFile(filePath: string) {
  const target = safePath(filePath);
  const stat = await fs.stat(target);

  if (stat.isDirectory()) {
    await fs.rmdir(target);
    return `Deleted directory ${relativePath(target)}`;
  }

  await fs.unlink(target);
  return `Deleted ${relativePath(target)}`;
}

async function searchFiles(query: string) {
  const results: string[] = [];

  async function walk(dir: string): Promise<void> {
    if (results.length >= 50) return;

    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      if (SKIP_DIRS.has(entry.name)) continue;

      const full = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        await walk(full);
        if (results.length >= 50) return;
        continue;
      }

      try {
        const text = await fs.readFile(full, "utf8");
        if (!text.includes(query)) continue;

        const lines = text
          .split(/\r?\n/)
          .map((line, index) =>
            line.includes(query) ? `${index + 1}: ${line.slice(0, 300)}` : "",
          )
          .filter(Boolean)
          .slice(0, 8);

        results.push(`${relativePath(full)}\n${lines.join("\n")}`);
      } catch {
        // Ignore binary/unreadable files.
      }

      if (results.length >= 50) return;
    }
  }

  await walk(ROOT);
  return results.join("\n\n") || "No matches found.";
}

async function runCommand(command: string): Promise<string> {
  const shell = process.env.ComSpec || "cmd.exe";

  const result = await new Promise<CommandResult>((resolve) => {
    const child = exec(command, {
      cwd: ROOT,
      shell,
      timeout: COMMAND_TIMEOUT,
      maxBuffer: 2_000_000,
      windowsHide: true,
    });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const finish = (value: CommandResult) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };

    child.stdout?.on("data", (chunk) => {
      stdout += String(chunk);
    });

    child.stderr?.on("data", (chunk) => {
      stderr += String(chunk);
    });

    child.on("error", (error) => {
      finish({
        stdout,
        stderr: `${stderr}${stderr ? "\n" : ""}${error.message}`,
        exitCode: 1,
      });
    });

    child.on("close", (code, signal) => {
      finish({
        stdout,
        stderr:
          signal && !stderr
            ? `Process terminated by signal: ${signal}`
            : stderr,
        exitCode: typeof code === "number" ? code : 1,
      });
    });
  });

  const sections = [
    `EXIT_CODE: ${result.exitCode}`,
    result.stdout ? `STDOUT:\n${result.stdout}` : "",
    result.stderr ? `STDERR:\n${result.stderr}` : "",
  ].filter(Boolean);

  const output = sections.join("\n\n").slice(0, MAX_COMMAND_OUTPUT);

  if (!output) {
    return result.exitCode === 0
      ? "Command completed with no output."
      : `Command failed with exit code ${result.exitCode}.`;
  }

  return output;
}

async function execute(call: ToolCall): Promise<string> {
  if (!call.name) throw new Error("Tool name is missing.");

  switch (call.name) {
    case "list_files":
      return listFiles(String(call.input.directory ?? "."));
    case "read_file":
      return readFile(String(call.input.file_path ?? ""));
    case "write_file":
      return writeFile(
        String(call.input.file_path ?? ""),
        String(call.input.content ?? ""),
      );
    case "edit_file":
      return editFile(
        String(call.input.file_path ?? ""),
        String(call.input.old_text ?? ""),
        String(call.input.new_text ?? ""),
      );
    case "delete_file":
      return deleteFile(String(call.input.file_path ?? ""));
    case "search_files":
      return searchFiles(String(call.input.query ?? ""));
    case "run_command":
      return runCommand(String(call.input.command ?? ""));
    default:
      throw new Error(`Unknown tool: ${call.name}`);
  }
}

async function executeWithRetry(call: ToolCall): Promise<string> {
  let lastError = "Unknown tool execution error.";

  for (let attempt = 1; attempt <= TOOL_RETRY_LIMIT + 1; attempt++) {
    try {
      return await execute(call);
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      if (attempt <= TOOL_RETRY_LIMIT) {
        console.log(`↻ Retrying ${call.name} (${attempt}/${TOOL_RETRY_LIMIT})`);
      }
    }
  }

  return `ERROR: ${lastError}`;
}

function normalizeToolArgs(value: unknown): Record<string, unknown> {
  if (value === undefined || value === null || value === "") return {};
  if (isRecord(value)) return value;

  if (typeof value !== "string") {
    throw new Error("Tool arguments must be a JSON object.");
  }

  const trimmed = value.trim();
  const withoutFence = trimmed
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();

  const parsed = JSON.parse(withoutFence);
  if (!isRecord(parsed)) {
    throw new Error("Tool arguments JSON must decode to an object.");
  }

  return parsed;
}

const openAiTools = tools.map((tool) => ({
  type: "function" as const,
  function: {
    name: tool.name,
    description: tool.description,
    parameters: tool.input_schema,
  },
}));

class GeminiProvider {
  private ai: any;

  constructor(key: string) {
    this.ai = new GoogleGenAI({ apiKey: key });
  }

  async run(userPrompt: string) {
    const history: any[] = [{ role: "user", parts: [{ text: userPrompt }] }];

    for (let step = 1; step <= MAX_STEPS; step++) {
      const response: any = await this.ai.models.generateContent({
        model: process.env.GEMINI_MODEL || "gemini-3.6-flash",
        contents: history,
        config: {
          systemInstruction: SYSTEM,
          tools: [
            {
              functionDeclarations: tools.map((tool) => ({
                name: tool.name,
                description: tool.description,
                parametersJsonSchema: tool.input_schema,
              })),
            },
          ],
        },
      });

      const parts = response.candidates?.[0]?.content?.parts ?? [];
      history.push({ role: "model", parts });

      for (const part of parts) {
        if (part.text) console.log(`\n${part.text}`);
      }

      const calls = parts
        .filter((part: any) => part.functionCall)
        .map((part: any) => part.functionCall);

      if (!calls.length) return response.text || "Done.";

      const functionResponses = [];

      for (const call of calls) {
        console.log(`\n→ ${call.name}`);

        const result = await executeWithRetry({
          id: call.id || call.name,
          name: call.name,
          input: isRecord(call.args) ? call.args : {},
        });

        console.log(
          result.startsWith("ERROR:")
            ? `✗ ${result}`
            : `✓ ${result.slice(0, 300)}`,
        );

        functionResponses.push({
          functionResponse: {
            name: call.name,
            response: { result },
          },
        });
      }

      history.push({ role: "user", parts: functionResponses });
    }

    throw new Error(
      `Agent stopped after ${MAX_STEPS} steps. Review the workspace and continue if needed.`,
    );
  }
}

class AnthropicProvider {
  private client: Anthropic;

  constructor(key: string) {
    this.client = new Anthropic({ apiKey: key });
  }

  async run(userPrompt: string) {
    const messages: any[] = [{ role: "user", content: userPrompt }];

    for (let step = 1; step <= MAX_STEPS; step++) {
      const response: any = await this.client.messages.create({
        model: process.env.ANTHROPIC_MODEL || "claude-sonnet-4-20250514",
        max_tokens: 12000,
        system: SYSTEM,
        tools: tools as any,
        messages,
      });

      const toolResults: ToolResult[] = [];

      for (const block of response.content) {
        if (block.type === "text") console.log(`\n${block.text}`);

        if (block.type === "tool_use") {
          console.log(`\n→ ${block.name}`);

          const result = await executeWithRetry({
            id: block.id,
            name: block.name,
            input: isRecord(block.input) ? block.input : {},
          });

          console.log(
            result.startsWith("ERROR:")
              ? `✗ ${result}`
              : `✓ ${result.slice(0, 300)}`,
          );

          toolResults.push({ id: block.id, result });
        }
      }

      messages.push({ role: "assistant", content: response.content });

      if (!toolResults.length) {
        return response.content
          .filter((block: any) => block.type === "text")
          .map((block: any) => block.text)
          .join("\n");
      }

      messages.push({
        role: "user",
        content: toolResults.map((result) => ({
          type: "tool_result",
          tool_use_id: result.id,
          content: result.result,
        })),
      });
    }

    throw new Error(
      `Agent stopped after ${MAX_STEPS} steps. Review the workspace and continue if needed.`,
    );
  }
}

class GroqProvider {
  private client: Groq;

  constructor(key: string) {
    this.client = new Groq({ apiKey: key });
  }

  async run(userPrompt: string) {
    const messages: any[] = [
      { role: "system", content: SYSTEM },
      { role: "user", content: userPrompt },
    ];

    for (let step = 1; step <= MAX_STEPS; step++) {
      const response: any =
        await this.client.chat.completions.create({
          model: process.env.GROQ_MODEL || "openai/gpt-oss-120b",
          messages,
          tools: openAiTools,
          tool_choice: "auto",
          temperature: 0,
        });

      const message = response.choices?.[0]?.message;
      if (!message) throw new Error("Groq returned no assistant message.");

      if (message.content) console.log(`\n${message.content}`);
      messages.push(message);

      const calls = message.tool_calls;
      if (!calls?.length) return message.content || "Done.";

      for (const call of calls) {
        console.log(`\n→ ${call.function.name}`);

        let result: string;

        try {
          const args = normalizeToolArgs(call.function.arguments);
          result = await executeWithRetry({
            id: call.id,
            name: call.function.name,
            input: args,
          });
        } catch (error) {
          result =
            `ERROR: Invalid tool arguments. ${
              error instanceof Error ? error.message : String(error)
            } Regenerate this tool call with valid JSON arguments matching the tool schema.`;
        }

        console.log(
          result.startsWith("ERROR:")
            ? `✗ ${result}`
            : `✓ ${result.slice(0, 300)}`,
        );

        messages.push({
          role: "tool",
          tool_call_id: call.id,
          content: result,
        });
      }
    }

    throw new Error(
      `Agent stopped after ${MAX_STEPS} steps. Review the workspace and continue if needed.`,
    );
  }
}

const rl = readline.createInterface({ input, output });

async function ask(question: string) {
  return (await rl.question(question)).trim();
}

async function main() {
  await fs.mkdir(ROOT, { recursive: true });
  console.log("\n=== Mini Coding Agent ===\n");
  console.log("1. Gemini (requires billing/access)");
  console.log("2. Anthropic (paid)");
  console.log("3. Groq (free, no billing required — recommended for testing)");

  const choice = await ask("Choose AI provider [1/2/3]: ");
  if (!["1", "2", "3"].includes(choice)) {
    throw new Error("Choose 1, 2, or 3.");
  }

  const envName =
    choice === "1"
      ? "GEMINI_API_KEY"
      : choice === "2"
        ? "ANTHROPIC_API_KEY"
        : "GROQ_API_KEY";

  const providerLabel =
    choice === "1" ? "Gemini" : choice === "2" ? "Anthropic" : "Groq";

  let key = process.env[envName];
  if (!key) key = await ask(`${providerLabel} API key: `);
  if (!key) throw new Error("An API key is required.");

  const request = await ask("\nWhat do you want to build?\n> ");
  if (!request) throw new Error("Describe the work you want done.");

  console.log(`\nWorkspace: ${ROOT}`);
  console.log("Starting agent...\n");

  const provider =
    choice === "1"
      ? new GeminiProvider(key)
      : choice === "2"
        ? new AnthropicProvider(key)
        : new GroqProvider(key);

  await provider.run(request);
  console.log("\n✓ Agent finished. Check the workspace for the completed project.\n");
}

main()
  .catch((error) => {
    console.error(
      `\n✗ ${error instanceof Error ? error.message : String(error)}\n`,
    );
    process.exitCode = 1;
  })
  .finally(() => rl.close());
