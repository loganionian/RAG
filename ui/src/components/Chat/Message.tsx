import { Bot, User } from 'lucide-react';
import type { SourceInfo } from '../../types/api';
import { SourceList } from './SourceList';

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceInfo[];
}

export function Message({ role, content, sources }: MessageProps) {
  const isUser = role === 'user';

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full ${
          isUser
            ? 'bg-primary-600 dark:bg-primary-500'
            : 'bg-gray-200 dark:bg-gray-700'
        }`}
      >
        {isUser ? (
          <User className="h-4 w-4 text-white" />
        ) : (
          <Bot className="h-4 w-4 text-gray-600 dark:text-gray-300" />
        )}
      </div>
      <div
        className={`max-w-[80%] rounded-xl px-4 py-3 ${
          isUser
            ? 'bg-primary-600 text-white dark:bg-primary-500'
            : 'card'
        }`}
      >
        <p className="whitespace-pre-wrap">{content}</p>
        {!isUser && sources && <SourceList sources={sources} />}
      </div>
    </div>
  );
}
