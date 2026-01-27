import { useEffect } from 'react';
import { Bot, ChevronDown, AlertCircle } from 'lucide-react';
import { useAgents } from '../../hooks/useAgents';
import type { AgentResponse } from '../../types/api';

interface AgentSelectorProps {
  selectedAgentId: string | null;
  onSelectAgent: (agent: AgentResponse | null) => void;
  disabled?: boolean;
}

export function AgentSelector({ selectedAgentId, onSelectAgent, disabled }: AgentSelectorProps) {
  const { agents, isLoading, error, fetchAgents } = useAgents();

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  // Auto-select default agent when agents load
  useEffect(() => {
    if (agents.length > 0 && selectedAgentId === null) {
      const defaultAgent = agents.find((a) => a.is_default) || agents[0];
      onSelectAgent(defaultAgent);
    }
  }, [agents, selectedAgentId, onSelectAgent]);

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const agentId = e.target.value;
    if (agentId === '') {
      onSelectAgent(null);
    } else {
      const agent = agents.find((a) => a.id === agentId);
      onSelectAgent(agent || null);
    }
  };

  if (error) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-600 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
        <AlertCircle className="h-4 w-4 flex-shrink-0" />
        <span>Failed to load agents</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <Bot className="h-4 w-4 text-gray-500 dark:text-gray-400" />
      <div className="relative">
        <select
          value={selectedAgentId || ''}
          onChange={handleChange}
          disabled={disabled || isLoading}
          className="appearance-none rounded-lg border border-gray-200 bg-white py-1.5 pl-3 pr-8 text-sm font-medium text-gray-700 transition-colors hover:border-gray-300 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200 dark:hover:border-gray-500"
        >
          {isLoading ? (
            <option value="">Loading agents...</option>
          ) : agents.length === 0 ? (
            <option value="">No agents available</option>
          ) : (
            agents.map((agent) => (
              <option key={agent.id} value={agent.id}>
                {agent.name}
                {agent.is_default ? ' (Default)' : ''}
              </option>
            ))
          )}
        </select>
        <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
      </div>
    </div>
  );
}
