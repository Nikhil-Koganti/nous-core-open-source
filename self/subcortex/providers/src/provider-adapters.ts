export * from './generated/provider-adapters.generated.js';

export { createAnthropicAdapter } from './adapters/anthropic-adapter.js';
export {
  createOllamaAdapter,
  isToolCapableModel,
} from './adapters/ollama-adapter.js';
