import { execFileSync, spawn } from 'node:child_process';
import { NousError, ValidationError } from '@nous/shared';
import type {
  IModelProvider,
  ModelProviderConfig,
  ModelRequest,
  ModelResponse,
  ModelStreamChunk,
  ProviderId,
} from '@nous/shared';
import {
  createAgentCliProviderAdapter,
  type AgentCliFailure,
  type AgentCliInvocation,
  type AgentCliInvocationDefaults,
  type AgentCliRawResult,
  type AgentCliRunResult,
  type AgentCliRunner,
  type AgentCliRunnerOptions,
  type AgentCliStreamEvent,
  normalizeAgentCliRunResult,
} from '../../protocols/agent-cli/index.js';
import { TextModelInputSchema, type TextModelInput } from '../../schemas/text-model-input.js';
import {
  QWEN_CODE_DEFAULT_MODEL_ID,
  QWEN_CODE_DEFAULT_TIMEOUT_MS,
  QWEN_CODE_MAX_TIMEOUT_MS,
  QWEN_CODE_PROVIDER_DEFINITION,
} from './definition.js';

export interface QwenCodeProviderOptions {
  readonly runner?: AgentCliRunner;
  readonly runnerOptions?: AgentCliRunnerOptions;
  readonly executable?: string;
}

export interface QwenCodeProcessRunnerOptions {
  readonly baseEnv?: Readonly<Record<string, string | undefined>>;
  readonly commandResolver?: QwenCodeCommandResolver;
  readonly platform?: NodeJS.Platform;
}

const QWEN_CODE_NOUS_BIN_ENV = 'NOUS_QWEN_CODE_BIN';
const QWEN_CODE_BIN_ENV = 'QWEN_CODE_BIN';

export type QwenCodeCommandResolver = (
  command: string,
  args: readonly string[],
  options: { readonly env?: NodeJS.ProcessEnv },
) => string;

export const QWEN_CODE_INVOCATION_DEFAULTS: AgentCliInvocationDefaults = {
  command: {
    executable: QWEN_CODE_PROVIDER_DEFINITION.agentCli.command.executable,
    defaultArgs: QWEN_CODE_PROVIDER_DEFINITION.agentCli.command.defaultArgs,
  },
  headless: {
    supported: QWEN_CODE_PROVIDER_DEFINITION.agentCli.headless.supported,
    requiredArgs: QWEN_CODE_PROVIDER_DEFINITION.agentCli.headless.requiredArgs,
    nonInteractiveEnv: QWEN_CODE_PROVIDER_DEFINITION.agentCli.headless.nonInteractiveEnv,
  },
  timeout: {
    defaultMs: QWEN_CODE_DEFAULT_TIMEOUT_MS,
    maxMs: QWEN_CODE_MAX_TIMEOUT_MS,
  },
};

export const QWEN_CODE_AGENT_ADAPTER = createAgentCliProviderAdapter({
  defaults: QWEN_CODE_INVOCATION_DEFAULTS,
});

export function createQwenCodeInvocationDefaults(
  executable: string = QWEN_CODE_PROVIDER_DEFINITION.agentCli.command.executable,
): AgentCliInvocationDefaults {
  return {
    ...QWEN_CODE_INVOCATION_DEFAULTS,
    command: {
      ...QWEN_CODE_INVOCATION_DEFAULTS.command,
      executable,
    },
  };
}

export class QwenCodeProvider implements IModelProvider {
  private readonly config: ModelProviderConfig;
  private readonly runner: AgentCliRunner;
  private readonly runnerOptions: AgentCliRunnerOptions | undefined;
  private readonly agentAdapter: typeof QWEN_CODE_AGENT_ADAPTER;

  constructor(config: ModelProviderConfig, options?: QwenCodeProviderOptions) {
    this.config = config;
    this.runner = options?.runner ?? createQwenCodeProcessRunner();
    this.runnerOptions = options?.runnerOptions;
    const executable = selectQwenCodeExecutable({
      explicitExecutable: options?.executable,
    });
    this.agentAdapter = createAgentCliProviderAdapter({
      defaults: createQwenCodeInvocationDefaults(executable),
    });
  }

  getConfig(): ModelProviderConfig {
    return this.config;
  }

  async invoke(request: ModelRequest): Promise<ModelResponse> {
    const runner = this.runner;
    const input = this.validateInput(request.input);
    const start = Date.now();
    const invocationInput = {
      args: this.invocationArgs(),
      input: renderTextModelInput(input),
      metadata: {
        provider: 'qwen-code',
        providerId: this.config.id,
        modelId: this.config.modelId,
        traceId: request.traceId,
      },
      runnerOptions: this.mergeRunnerOptions(request),
    };

    const result = await this.agentAdapter.invoke(invocationInput, runner);

    if (!result.ok) {
      throw toProviderError(result.failure, result.stderr, result.stdout);
    }

    return {
      output: result.stdout,
      providerId: this.config.id as ProviderId,
      usage: {
        computeMs: result.durationMs ?? Date.now() - start,
      },
      traceId: request.traceId,
    };
  }

  async *stream(request: ModelRequest): AsyncIterable<ModelStreamChunk> {
    const runner = this.runner;
    const input = this.validateInput(request.input);
    const state: QwenCodeStreamState = {
      emittedContent: false,
      finalResult: undefined,
    };

    const invocationInput = {
      args: this.invocationArgs(),
      input: renderTextModelInput(input),
      metadata: {
        provider: 'qwen-code',
        providerId: this.config.id,
        modelId: this.config.modelId,
        traceId: request.traceId,
      },
      runnerOptions: this.mergeRunnerOptions(request),
    };

    try {
      for await (const event of this.agentAdapter.stream(invocationInput, runner)) {
        if (event.stream === 'system') {
          state.finalResult = event.result;
          continue;
        }

        if (event.stream !== 'stdout') continue;

        if (event.text.length > 0) {
          state.emittedContent = true;
          yield { content: event.text, done: false };
        }
      }

      if (state.finalResult === undefined) {
        throw new NousError(
          'Qwen Code streaming ended without a process result.',
          'PROVIDER_UNAVAILABLE',
          { provider: 'qwen-code', failureKind: 'unknown' },
        );
      }

      if (!state.finalResult.ok) {
        throw toProviderError(
          state.finalResult.failure,
          state.finalResult.stderr,
          state.finalResult.stdout,
        );
      }

      if (!state.emittedContent && state.finalResult.stdout.length > 0) {
        yield { content: state.finalResult.stdout, done: false };
      }

      yield { content: '', done: true };
    } catch (error) {
      if (error instanceof NousError) throw error;
      throw new NousError(
        `Qwen Code streaming failed: ${error instanceof Error ? error.message : String(error)}`,
        'PROVIDER_UNAVAILABLE',
        { provider: 'qwen-code', failureKind: 'unknown' },
      );
    }
  }

  private validateInput(input: unknown): TextModelInput {
    const result = TextModelInputSchema.safeParse(input);
    if (!result.success) {
      throw new ValidationError(
        'Invalid Qwen Code provider input',
        result.error.errors.map((error) => ({
          path: error.path.join('.'),
          message: error.message,
        })),
      );
    }
    return result.data;
  }

  private invocationArgs(): readonly string[] {
    // Qwen Code headless mode reads the prompt from stdin, so the rendered
    // prompt is piped via `invocation.input` rather than passed as a flag value.
    if (
      this.config.modelId.length === 0 ||
      this.config.modelId === QWEN_CODE_DEFAULT_MODEL_ID
    ) {
      return [];
    }

    return ['--model', this.config.modelId];
  }

  private mergeRunnerOptions(request: ModelRequest): AgentCliRunnerOptions | undefined {
    const signal = request.abortSignal
      ? { aborted: request.abortSignal.aborted }
      : this.runnerOptions?.signal;

    if (!signal && !this.runnerOptions) {
      return undefined;
    }

    return {
      ...this.runnerOptions,
      ...(signal ? { signal } : {}),
    };
  }
}

export function selectQwenCodeExecutable(
  options: {
    readonly explicitExecutable?: string;
    readonly env?: Readonly<Record<string, string | undefined>>;
  } = {},
): string {
  const env = options.env ?? process.env;
  return options.explicitExecutable
    ?? env[QWEN_CODE_NOUS_BIN_ENV]
    ?? env[QWEN_CODE_BIN_ENV]
    ?? QWEN_CODE_PROVIDER_DEFINITION.agentCli.command.executable;
}

export function resolveQwenCodeExecutable(
  executable: string = QWEN_CODE_PROVIDER_DEFINITION.agentCli.command.executable,
  options: {
    readonly commandResolver?: QwenCodeCommandResolver;
    readonly env?: Readonly<Record<string, string | undefined>>;
    readonly platform?: NodeJS.Platform;
  } = {},
): string {
  if (executable !== QWEN_CODE_PROVIDER_DEFINITION.agentCli.command.executable) {
    return executable;
  }

  const platform = options.platform ?? process.platform;
  const candidates = platform === 'win32'
    ? resolveExecutableCandidates('where.exe', ['qwen'], options)
    : resolveExecutableCandidates('which', ['-a', 'qwen'], options);
  const systemCandidates = candidates.filter((candidate) => !isNodeModulesBinCandidate(candidate));

  if (platform === 'win32') {
    return systemCandidates.find((candidate) => candidate.toLowerCase().endsWith('.cmd'))
      ?? systemCandidates[0]
      ?? executable;
  }

  return systemCandidates[0] ?? executable;
}

function resolveExecutableCandidates(
  command: string,
  args: readonly string[],
  options: {
    readonly commandResolver?: QwenCodeCommandResolver;
    readonly env?: Readonly<Record<string, string | undefined>>;
  },
): readonly string[] {
  const commandResolver = options.commandResolver ?? defaultCommandResolver;
  try {
    return commandResolver(command, args, { env: toProcessEnv(options.env) })
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
  } catch {
    return [];
  }
}

function defaultCommandResolver(
  command: string,
  args: readonly string[],
  options: { readonly env?: NodeJS.ProcessEnv },
): string {
  return execFileSync(command, [...args], {
    encoding: 'utf8',
    env: options.env,
  });
}

function isNodeModulesBinCandidate(candidate: string): boolean {
  return candidate.replace(/\\/g, '/').toLowerCase().includes('/node_modules/.bin/');
}

function toProcessEnv(env: Readonly<Record<string, string | undefined>> | undefined): NodeJS.ProcessEnv | undefined {
  if (!env) return undefined;
  const processEnv: NodeJS.ProcessEnv = {};
  assignDefinedEnv(processEnv, env);
  return processEnv;
}

interface QwenCodeStreamState {
  emittedContent: boolean;
  finalResult: AgentCliRunResult | undefined;
}

export function createQwenCodeProcessRunner(
  options: QwenCodeProcessRunnerOptions = {},
): AgentCliRunner {
  return {
    async run(invocation, runnerOptions) {
      return normalizeAgentCliRunResult(
        await runQwenCodeProcessRaw(invocation, runnerOptions, options),
      );
    },
    async *stream(invocation, runnerOptions) {
      const queue: AgentCliStreamEvent[] = [];
      let wake: (() => void) | undefined;
      let completed = false;

      const push = (event: AgentCliStreamEvent): void => {
        queue.push(event);
        wake?.();
        wake = undefined;
      };

      const waitForEvent = (): Promise<void> => {
        if (queue.length > 0 || completed) return Promise.resolve();
        return new Promise((resolve) => {
          wake = resolve;
        });
      };

      void runQwenCodeProcessRaw(invocation, runnerOptions, options, (event) => {
        push(event);
      }).then((rawResult) => {
        push({ stream: 'system', result: normalizeAgentCliRunResult(rawResult) });
      }).finally(() => {
        completed = true;
        wake?.();
        wake = undefined;
      });

      while (!completed || queue.length > 0) {
        await waitForEvent();
        while (queue.length > 0) {
          yield queue.shift()!;
        }
      }
    },
  };
}

function runQwenCodeProcessRaw(
  invocation: AgentCliInvocation,
  runnerOptions: AgentCliRunnerOptions | undefined,
  options: QwenCodeProcessRunnerOptions,
  onTranscriptEvent?: (event: AgentCliStreamEvent) => void,
): Promise<AgentCliRawResult> {
  const startedAt = Date.now();
  if (runnerOptions?.signal?.aborted) {
    return Promise.resolve({
      startedAt,
      endedAt: Date.now(),
      error: new Error('Agent CLI invocation aborted before start.'),
    });
  }

  return new Promise((resolve) => {
    let child;
    try {
      const env = buildProcessEnv(invocation.command.env, runnerOptions, options);
      child = spawn(resolveSpawnExecutable(invocation.command.executable, {
        commandResolver: options.commandResolver,
        env,
        platform: options.platform,
      }), [...(invocation.command.args ?? [])], {
        cwd: invocation.command.cwd,
        env,
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true,
        shell: process.platform === 'win32',
      });
    } catch (error) {
      resolve({
        error,
        startedAt,
        endedAt: Date.now(),
      });
      return;
    }

    let stdout = '';
    let stderr = '';
    let timedOut = false;
    let settled = false;
    const timeout = invocation.timeoutMs === undefined
      ? undefined
      : setTimeout(() => {
        timedOut = true;
        child.kill('SIGTERM');
      }, invocation.timeoutMs);

    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk: string) => {
      stdout += chunk;
      onTranscriptEvent?.({ stream: 'stdout', text: chunk, timestamp: new Date().toISOString() });
    });
    child.stderr.on('data', (chunk: string) => {
      stderr += chunk;
      onTranscriptEvent?.({ stream: 'stderr', text: chunk, timestamp: new Date().toISOString() });
    });
    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      if (timeout) clearTimeout(timeout);
      resolve({
        stdout,
        stderr,
        error,
        timedOut,
        startedAt,
        endedAt: Date.now(),
      });
    });
    child.on('close', (exitCode, signal) => {
      if (settled) return;
      settled = true;
      if (timeout) clearTimeout(timeout);
      resolve({
        exitCode,
        signal,
        stdout,
        stderr,
        timedOut,
        startedAt,
        endedAt: Date.now(),
      });
    });

    child.stdin.end(invocation.input ?? '');
  });
}

function resolveSpawnExecutable(
  executable: string,
  options: {
    readonly commandResolver?: QwenCodeCommandResolver;
    readonly env?: Readonly<Record<string, string | undefined>>;
    readonly platform?: NodeJS.Platform;
  },
): string {
  return resolveQwenCodeExecutable(executable, options);
}

function buildProcessEnv(
  invocationEnv: Readonly<Record<string, string>> | undefined,
  runnerOptions: AgentCliRunnerOptions | undefined,
  options: QwenCodeProcessRunnerOptions,
): NodeJS.ProcessEnv {
  const baseEnv = options.baseEnv ?? process.env;
  const policy = runnerOptions?.environmentPolicy;
  const env: NodeJS.ProcessEnv = {};

  if (!policy || policy.mergeStrategy === 'explicit') {
    assignDefinedEnv(env, baseEnv);
  } else if (policy.mergeStrategy === 'allowlist') {
    for (const key of policy.allowlist ?? []) {
      const value = baseEnv[key];
      if (value !== undefined && isValidEnvKey(key)) env[key] = value;
    }
  }

  assignDefinedEnv(env, policy?.env);
  assignDefinedEnv(env, invocationEnv);
  return env;
}

function assignDefinedEnv(
  target: NodeJS.ProcessEnv,
  source: Readonly<Record<string, string | undefined>> | undefined,
): void {
  if (!source) return;
  for (const [key, value] of Object.entries(source)) {
    if (value !== undefined && isValidEnvKey(key)) {
      target[key] = value;
    }
  }
}

function isValidEnvKey(key: string): boolean {
  return key.length > 0 && !key.includes('=');
}

function renderTextModelInput(input: TextModelInput): string {
  if ('prompt' in input) {
    return input.prompt;
  }

  const sections: string[] = [];
  if (typeof input.system === 'string' && input.system.trim().length > 0) {
    sections.push(input.system.trim());
  } else if (Array.isArray(input.system) && input.system.length > 0) {
    sections.push(JSON.stringify(input.system, null, 2));
  }

  for (const message of input.messages) {
    sections.push(`${message.role}: ${renderMessageContent(message.content)}`);
  }

  if (input.tools && input.tools.length > 0) {
    sections.push(`Available tools:\n${JSON.stringify(input.tools, null, 2)}`);
  }

  return sections.join('\n\n');
}

function renderMessageContent(content: string | readonly unknown[]): string {
  return typeof content === 'string'
    ? content
    : JSON.stringify(content, null, 2);
}

function toProviderError(
  failure: AgentCliFailure | undefined,
  stderr?: string,
  stdout?: string,
): NousError {
  if (!failure) {
    return new NousError(
      'Qwen Code invocation failed.',
      'PROVIDER_UNAVAILABLE',
      { provider: 'qwen-code', stderr, stdout },
    );
  }

  return new NousError(
    stderr && stderr.trim().length > 0
      ? `${failure.message} ${stderr.trim().slice(0, 500)}`
      : failure.message,
    failure.kind === 'auth' ? 'PROVIDER_ERROR' : 'PROVIDER_UNAVAILABLE',
    {
      provider: 'qwen-code',
      failureKind: failure.kind,
      exitCode: failure.exitCode,
      signal: failure.signal,
      timedOut: failure.timedOut,
      stderr,
      stdout,
    },
  );
}
