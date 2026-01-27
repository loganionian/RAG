import { Bot } from 'lucide-react';
import type { AgentResponse } from '../../types/api';

interface ChatHeaderProps {
  agent: AgentResponse | null;
}

export function ChatHeader({ agent }: ChatHeaderProps) {
  if (!agent) {
    return null;
  }

  return (
    <div className="flex items-center gap-2 border-b border-gray-200 bg-gray-50 px-4 py-2 dark:border-gray-700 dark:bg-gray-800/50">
      <Bot className="h-4 w-4 text-primary-500" />
      <span className="text-sm font-medium text-gray-700 dark:text-gray-200">
        {agent.name}
      </span>
      {agent.description && (
        <span className="text-sm text-gray-500 dark:text-gray-400">
          — {agent.description}
        </span>
      )}
    </div>
  );
}
