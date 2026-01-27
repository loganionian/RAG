import { Bot, Pencil, Trash2, Star } from 'lucide-react';
import type { AgentResponse } from '../../types/api';

interface AgentRowProps {
  agent: AgentResponse;
  onEdit: (agent: AgentResponse) => void;
  onDelete: (agent: AgentResponse) => void;
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

export function AgentRow({ agent, onEdit, onDelete }: AgentRowProps) {
  return (
    <tr className="border-b border-gray-200 dark:border-gray-700">
      <td className="py-3 pr-4">
        <div className="flex items-center gap-3">
          <Bot className="h-5 w-5 flex-shrink-0 text-gray-400" />
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="truncate font-medium text-gray-900 dark:text-gray-100">
                {agent.name}
              </span>
              {agent.is_default && (
                <span className="inline-flex items-center gap-1 rounded-full bg-yellow-100 px-2 py-0.5 text-xs font-medium text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400">
                  <Star className="h-3 w-3" />
                  Default
                </span>
              )}
            </div>
            {agent.description && (
              <p className="truncate text-sm text-gray-500 dark:text-gray-400">
                {agent.description}
              </p>
            )}
          </div>
        </div>
      </td>
      <td className="py-3 pr-4">
        <span className="text-gray-600 dark:text-gray-400">
          {agent.temperature !== null ? agent.temperature : '-'}
        </span>
      </td>
      <td className="py-3 pr-4">
        <span className="text-gray-600 dark:text-gray-400">
          {agent.max_tokens !== null ? agent.max_tokens.toLocaleString() : '-'}
        </span>
      </td>
      <td className="py-3 pr-4">
        <span className="text-gray-500 dark:text-gray-400">
          {formatDate(agent.updated_at)}
        </span>
      </td>
      <td className="py-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => onEdit(agent)}
            className="rounded p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:hover:bg-gray-700 dark:hover:text-gray-300"
            aria-label={`Edit ${agent.name}`}
          >
            <Pencil className="h-4 w-4" />
          </button>
          <button
            onClick={() => onDelete(agent)}
            disabled={agent.is_default}
            className="rounded p-1.5 text-gray-400 hover:bg-red-50 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-gray-400 dark:hover:bg-red-900/30 dark:hover:text-red-400"
            aria-label={`Delete ${agent.name}`}
            title={agent.is_default ? 'Cannot delete default agent' : undefined}
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </td>
    </tr>
  );
}
