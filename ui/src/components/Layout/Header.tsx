import { Bot, Circle } from 'lucide-react';

interface HeaderProps {
  isConnected: boolean;
  isChecking?: boolean;
}

export function Header({ isConnected, isChecking }: HeaderProps) {
  return (
    <header className="flex h-14 items-center justify-between border-b border-gray-200 bg-white px-4 dark:border-gray-700 dark:bg-gray-800">
      <div className="flex items-center gap-2">
        <Bot className="h-6 w-6 text-primary-600 dark:text-primary-400" />
        <h1 className="text-lg font-semibold text-gray-900 dark:text-gray-100">RAG Demo</h1>
      </div>
      <div className="flex items-center gap-2">
        <Circle
          className={`h-3 w-3 ${
            isChecking
              ? 'animate-pulse fill-yellow-500 text-yellow-500'
              : isConnected
                ? 'fill-green-500 text-green-500'
                : 'fill-red-500 text-red-500'
          }`}
        />
        <span className="text-sm text-gray-600 dark:text-gray-400">
          {isChecking ? 'Checking...' : isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
    </header>
  );
}
