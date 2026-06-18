import { spawn } from 'node:child_process';
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
  normalizeAgentCliRunResult,
  type AgentCliFailure,
  type AgentCliInvocation,
  type AgentCliInvocationDefaults,
  type AgentCliRawResult,
  type AgentCliRunner,
  type AgentCliRunnerOptions,
} from '../../protocols/agent-cli/index.js';
import { TextModelInputSchema, type TextModelInput } from '../../schemas/text-model-input.js';
import type { ProviderFactoryModule } from '../../schemas/provider-factory.js';
import {
  providerAdapter,
  renderGhCopilotPrompt,
} from './adapter.js';
import {
  GITHUB_COPILOT_CLI_DEFAULT_TIMEOUT_MS,
  GITHUB_COPILOT_CLI_MAX_TIMEOUT_MS,
  GITHUB_COPILOT_CLI_PROVIDER_DEFINITION,
} from './definition.js';

export const GITHUB_COPILOT_CLI_INVOCATION_DEFAULTS: AgentCliInvocationDefaults = {
  command: {
    executable: GITHUB_COPILOT_CLI_PROVIDER_DEFINITION.agentCli.command.executable,
    defaultArgs: GITHUB_COPILOT_CLI_PROVIDER_DEFINITION.agentCli.command.defaultArgs,
  },
  headless: {
    supported: GITHUB_COPILOT_CLI_PROVIDER_DEFINITION.agentCli.headless.supported,
    requiredArgs: GITHUB_COPILOT_CLI_PROVIDER_DEFINITION.agentCli.headless.requiredArgs,
    nonInteractiveEnv: GITHUB_COPILOT_CLI_PROVIDER_DEFINITION.agentCli.headless.nonInteractiveEnv,
  },
  timeout: {
    defaultMs: GITHUB_COPILOT_CLI_DEFAULT_TIMEOUT_MS,
    maxMs: GITHUB_COPILOT_CLI_MAX_TIMEOUT_MS,
  },
};

export const GITHUB_COPILOT_CLI_AGENT_ADAPTER = createAgentCliProviderAdapter({
  defaults: GITHUB_COPILOT_CLI_INVOCATION_DEFAULTS,
});

export function createGhProcessRunner(): AgentCliRunner {
  return {
    async run(invocation: AgentCliInvocation, runnerOptions?: AgentCliRunnerOptions) {
      return normalizeAgentCliRunResult(
        await runGhProcessRaw(invocation, runnerOptions),
      );
    },
  };
}

function runGhProcessRaw(
  invocation: AgentCliInvocation,
  runnerOptions: AgentCliRunnerOptions | undefined,
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
      const spawnEnv: NodeJS.ProcessEnv = Object.assign(
        {},
        process.env,
        invocation.command.env ?? {},
      );
      child = spawn(
        invocation.command.executable,
        [...(invocation.command.args ?? [])],
        {
          cwd: invocation.command.cwd,
          env: spawnEnv,
          stdio: ['ignore', 'pipe', 'pipe'],
          windowsHide: true,
        },
      );
    } catch (error) {
      resolve({ error, startedAt, endedAt: Date.now() });
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

    if (child.stdout) {
      child.stdout.setEncoding('utf8');
      child.stdout.on('data', (chunk: string) => { stdout += chunk; });
    }
    if (child.stderr) {
      child.stderr.setEncoding('utf8');
      child.stderr.on('data', (chunk: string) => { stderr += chunk; });
    }

    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      if (timeout !== undefined) clearTimeout(timeout);
      resolve({ stdout, stderr, error, timedOut, startedAt, endedAt: Date.now() });
    });

    child.on('close', (exitCode, signal) => {
      if (settled) return;
      settled = true;
      if (timeout !== undefined) clearTimeout(timeout);
      resolve({ exitCode, signal, stdout, stderr, timedOut, startedAt, endedAt: Date.now() });
    });
  });
}

export class GitHubCopilotCliProvider implements IModelProvider {
  private readonly config: ModelProviderConfig;
  private readonly runner: AgentCliRunner;
  private readonly runnerOptions: AgentCliRunnerOptions | undefined;

  constructor(
    config: ModelProviderConfig,
    options?: { runner?: AgentCliRunner; runnerOptions?: AgentCliRunnerOptions },
  ) {
    this.config = config;
    this.runner = options?.runner ?? createGhProcessRunner();
    this.runnerOptions = options?.runnerOptions;
  }

  getConfig(): ModelProviderConfig {
    return this.config;
  }

  async invoke(request: ModelRequest): Promise<ModelResponse> {
    const input = this.validateInput(request.input);
    const promptString = formatModelInput(input);
    const start = Date.now();

    const result = await GITHUB_COPILOT_CLI_AGENT_ADAPTER.invoke(
      {
        args: [promptString],
        metadata: {
          provider: 'github-copilot-cli',
          providerId: this.config.id,
          modelId: this.config.modelId,
          traceId: request.traceId,
        },
        runnerOptions: this.mergeRunnerOptions(request),
      },
      this.runner,
    );

    if (!result.ok) {
      throw toProviderError(result.failure, result.stderr, result.stdout);
    }

    const parsed = providerAdapter.create().parseResponse(result.stdout, request.traceId);

    return {
      output: parsed.response,
      providerId: this.config.id as ProviderId,
      usage: {
        computeMs: result.durationMs ?? Date.now() - start,
      },
      traceId: request.traceId,
    };
  }

  async *stream(request: ModelRequest): AsyncIterable<ModelStreamChunk> {
    // gh copilot suggest does not support streaming — invoke and yield one chunk.
    const response = await this.invoke(request);
    const content = typeof response.output === 'string'
      ? response.output
      : String(response.output ?? '');
    yield { content, done: false };
    yield { content: '', done: true };
  }

  private validateInput(input: unknown): TextModelInput {
    const result = TextModelInputSchema.safeParse(input);
    if (!result.success) {
      throw new ValidationError(
        'Invalid GitHub Copilot CLI provider input',
        result.error.errors.map((error) => ({
          path: error.path.join('.'),
          message: error.message,
        })),
      );
    }
    return result.data;
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

function formatModelInput(input: TextModelInput): string {
  if ('prompt' in input) {
    return input.prompt;
  }
  const system = typeof input.system === 'string'
    ? input.system
    : Array.isArray(input.system)
      ? input.system.map(String).join('\n\n')
      : '';
  return renderGhCopilotPrompt(system, input.messages, undefined);
}

function toProviderError(
  failure: AgentCliFailure | undefined,
  stderr?: string,
  stdout?: string,
): NousError {
  if (!failure) {
    return new NousError(
      '[github-copilot-cli] invocation failed',
      'PROVIDER_UNAVAILABLE',
      { provider: 'github-copilot-cli', stderr, stdout },
    );
  }

  return new NousError(
    stderr && stderr.trim().length > 0
      ? `[github-copilot-cli] ${failure.message} ${stderr.trim().slice(0, 500)}`
      : `[github-copilot-cli] ${failure.message}`,
    failure.kind === 'auth' ? 'PROVIDER_ERROR' : 'PROVIDER_UNAVAILABLE',
    {
      provider: 'github-copilot-cli',
      failureKind: failure.kind,
      exitCode: failure.exitCode,
      signal: failure.signal,
      timedOut: failure.timedOut,
      stderr,
      stdout,
    },
  );
}

export const providerFactory = {
  vendorKey: 'github-copilot-cli',
  create(config, options) {
    return new GitHubCopilotCliProvider(config, {
      runner: options?.agentCliRunner,
      runnerOptions: options?.agentCliRunnerOptions,
    });
  },
} as const satisfies ProviderFactoryModule;
