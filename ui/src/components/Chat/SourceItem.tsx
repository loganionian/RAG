import { FileText } from 'lucide-react';
import type { SourceInfo } from '../../types/api';

interface SourceItemProps {
  source: SourceInfo;
}

export function SourceItem({ source }: SourceItemProps) {
  const truncatedSnippet =
    source.snippet.length > 150 ? `${source.snippet.slice(0, 150)}...` : source.snippet;

  return (
    <div className="flex gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3 dark:border-gray-600 dark:bg-gray-700/50">
      <FileText className="mt-0.5 h-4 w-4 flex-shrink-0 text-gray-400" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate font-medium text-gray-900 dark:text-gray-100">
            {source.filename}
          </span>
          {source.page !== null && (
            <span className="flex-shrink-0 rounded bg-gray-200 px-1.5 py-0.5 text-xs text-gray-600 dark:bg-gray-600 dark:text-gray-300">
              Page {source.page}
            </span>
          )}
        </div>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">{truncatedSnippet}</p>
      </div>
    </div>
  );
}
