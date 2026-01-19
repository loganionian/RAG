export function LoadingIndicator() {
  return (
    <div className="flex items-center gap-1" role="status" aria-label="Loading">
      <span className="sr-only">Thinking...</span>
      <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400 dark:bg-gray-500 [animation-delay:-0.3s]" />
      <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400 dark:bg-gray-500 [animation-delay:-0.15s]" />
      <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400 dark:bg-gray-500" />
    </div>
  );
}
