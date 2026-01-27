import { RefreshCw, Bot, Plus } from 'lucide-react';
import type { AgentResponse } from '../../types/api';
import { AgentRow } from './AgentRow';

interface AgentListProps {
  agents: AgentResponse[];
  isLoading: boolean;
  onRefresh: () => void;
  onCreate: () => void;
  onEdit: (agent: AgentResponse) => void;
  onDelete: (agent: AgentResponse) => void;
}

export function AgentList({
  agents,
  isLoading,
  onRefresh,
  onCreate,
  onEdit,
  onDelete,
}: AgentListProps) {
  if (agents.length === 0 && !isLoading) {
    return (
      <div className="card flex flex-col items-center justify-center py-12">
        <Bot className="h-12 w-12 text-gray-300 dark:text-gray-600" />
        <p className="mt-3 text-gray-600 dark:text-gray-400">No agents configured yet</p>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-500">
          Create an agent to customize LLM behavior
        </p>
        <button onClick={onCreate} className="btn btn-primary mt-4">
          <Plus className="mr-2 h-4 w-4" />
          Create Agent
        </button>
      </div>
    );
  }

  return (
    <div className="card overflow-hidden p-0">
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3 dark:border-gray-700">
        <h3 className="font-medium text-gray-900 dark:text-gray-100">
          Agents ({agents.length})
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="btn btn-secondary p-2"
            aria-label="Refresh agents"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
          <button onClick={onCreate} className="btn btn-primary">
            <Plus className="mr-2 h-4 w-4" />
            Create Agent
          </button>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-200 bg-gray-50 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:border-gray-700 dark:bg-gray-800/50 dark:text-gray-400">
              <th className="px-4 py-3">Name</th>
              <th className="px-4 py-3">Temperature</th>
              <th className="px-4 py-3">Max Tokens</th>
              <th className="px-4 py-3">Updated</th>
              <th className="px-4 py-3">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {agents.map((agent) => (
              <AgentRow
                key={agent.id}
                agent={agent}
                onEdit={onEdit}
                onDelete={onDelete}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
