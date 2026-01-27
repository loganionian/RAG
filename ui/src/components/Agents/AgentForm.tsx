import { useState, useEffect } from 'react';
import type { AgentCreate, AgentResponse, AgentUpdate } from '../../types/api';

interface FormErrors {
  name?: string;
  system_prompt?: string;
  temperature?: string;
  max_tokens?: string;
}

interface AgentFormProps {
  agent?: AgentResponse | null;
  onSubmit: (data: AgentCreate | AgentUpdate) => void;
  onCancel: () => void;
  isLoading?: boolean;
}

export function AgentForm({ agent, onSubmit, onCancel, isLoading = false }: AgentFormProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [temperature, setTemperature] = useState('');
  const [maxTokens, setMaxTokens] = useState('');
  const [errors, setErrors] = useState<FormErrors>({});

  const isEditing = !!agent;

  useEffect(() => {
    if (agent) {
      setName(agent.name);
      setDescription(agent.description || '');
      setSystemPrompt(agent.system_prompt);
      setTemperature(agent.temperature !== null ? String(agent.temperature) : '');
      setMaxTokens(agent.max_tokens !== null ? String(agent.max_tokens) : '');
    }
  }, [agent]);

  const validate = (): boolean => {
    const newErrors: FormErrors = {};

    if (!name.trim()) {
      newErrors.name = 'Name is required';
    } else if (name.length > 100) {
      newErrors.name = 'Name must be 100 characters or less';
    }

    if (!systemPrompt.trim()) {
      newErrors.system_prompt = 'System prompt is required';
    }

    if (temperature) {
      const temp = parseFloat(temperature);
      if (isNaN(temp) || temp < 0 || temp > 2) {
        newErrors.temperature = 'Temperature must be between 0 and 2';
      }
    }

    if (maxTokens) {
      const tokens = parseInt(maxTokens, 10);
      if (isNaN(tokens) || tokens < 1 || tokens > 32000) {
        newErrors.max_tokens = 'Max tokens must be between 1 and 32000';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!validate()) {
      return;
    }

    const data: AgentCreate | AgentUpdate = {
      name: name.trim(),
      description: description.trim() || undefined,
      system_prompt: systemPrompt.trim(),
      temperature: temperature ? parseFloat(temperature) : null,
      max_tokens: maxTokens ? parseInt(maxTokens, 10) : null,
    };

    onSubmit(data);
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      {/* Name */}
      <div>
        <label
          htmlFor="agent-name"
          className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Name <span className="text-red-500">*</span>
        </label>
        <input
          id="agent-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className={`input w-full ${errors.name ? 'border-red-500' : ''}`}
          placeholder="e.g., Research Assistant"
          maxLength={100}
        />
        {errors.name && (
          <p className="mt-1 text-sm text-red-500">{errors.name}</p>
        )}
      </div>

      {/* Description */}
      <div>
        <label
          htmlFor="agent-description"
          className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Description
        </label>
        <input
          id="agent-description"
          type="text"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          className="input w-full"
          placeholder="Brief description of the agent's purpose"
        />
      </div>

      {/* System Prompt */}
      <div>
        <label
          htmlFor="agent-system-prompt"
          className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          System Prompt <span className="text-red-500">*</span>
        </label>
        <textarea
          id="agent-system-prompt"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          rows={5}
          className={`input w-full resize-y ${errors.system_prompt ? 'border-red-500' : ''}`}
          placeholder="You are a helpful assistant that..."
        />
        {errors.system_prompt && (
          <p className="mt-1 text-sm text-red-500">{errors.system_prompt}</p>
        )}
      </div>

      {/* Temperature & Max Tokens */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label
            htmlFor="agent-temperature"
            className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Temperature
          </label>
          <input
            id="agent-temperature"
            type="number"
            step="0.1"
            min="0"
            max="2"
            value={temperature}
            onChange={(e) => setTemperature(e.target.value)}
            className={`input w-full ${errors.temperature ? 'border-red-500' : ''}`}
            placeholder="0.7"
          />
          {errors.temperature && (
            <p className="mt-1 text-sm text-red-500">{errors.temperature}</p>
          )}
        </div>

        <div>
          <label
            htmlFor="agent-max-tokens"
            className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Max Tokens
          </label>
          <input
            id="agent-max-tokens"
            type="number"
            min="1"
            max="32000"
            value={maxTokens}
            onChange={(e) => setMaxTokens(e.target.value)}
            className={`input w-full ${errors.max_tokens ? 'border-red-500' : ''}`}
            placeholder="1000"
          />
          {errors.max_tokens && (
            <p className="mt-1 text-sm text-red-500">{errors.max_tokens}</p>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="mt-2 flex justify-end gap-3">
        <button
          type="button"
          onClick={onCancel}
          disabled={isLoading}
          className="btn btn-secondary"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={isLoading}
          className="btn btn-primary"
        >
          {isLoading ? 'Saving...' : isEditing ? 'Save Changes' : 'Create Agent'}
        </button>
      </div>
    </form>
  );
}
