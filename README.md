# Mini Coding Agent

A small, independent autonomous coding-agent prototype. It accepts a natural-language coding request, uses an AI provider to plan and act, edits files, runs commands, verifies the result, and fixes failures inside an isolated local workspace.

## Providers

- **Groq** — recommended for free testing
- **Gemini** — requires API access/billing depending on the Google project
- **Anthropic** — requires an Anthropic API key

## What it does

1. Starts in your terminal.
2. Lets you choose an AI provider.
3. Prompts for the API key if it is not already available in the environment.
4. Accepts a natural-language coding request.
5. Reads, creates, edits, and searches files inside `workspace/`.
6. Runs Windows shell commands for installs, builds, tests, and debugging.
7. Continues through tool calls until the requested work is complete.
8. Audits the project and reports verification failures instead of blindly claiming success.

## Requirements

- Node.js 20+
- A Groq, Gemini, or Anthropic API key
- Windows is currently the primary local development environment

## Install

```bash
npm install
npm run build
npm start
```

For development:

```bash
npm run dev
```

The agent automatically creates `workspace/`. Generated projects live there and are ignored by Git.

## Environment variables

Copy `.env.example` to `.env` if you want to keep a provider key in your local environment, or simply enter the key when prompted.

```text
GROQ_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
GROQ_MODEL=openai/gpt-oss-120b
GEMINI_MODEL=gemini-3.6-flash
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

Never commit real API keys.

## Safety / architecture note

This is a local Section-1 prototype. Agent commands execute on the same machine running the process. Before turning this into a multi-user hosted service, arbitrary generated code must run inside isolated containers or sandboxes with resource limits and project-level filesystem isolation.

## Roadmap

- Browser chat UI
- Live preview
- Hosted agent workers
- Isolated per-project sandboxes
- GitHub project synchronization
- Deployment/runtime management
- Accounts, usage limits, billing, and moderation
