import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import type { SourceInfo } from '../../types/api';
import { SourceItem } from './SourceItem';

interface SourceListProps {
  sources: SourceInfo[];
}

export function SourceList({ sources }: SourceListProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-3">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
      >
        <ChevronDown
          className={`h-4 w-4 transition-transform ${isExpanded ? 'rotate-0' : '-rotate-90'}`}
        />
        <span>Sources ({sources.length})</span>
      </button>
      {isExpanded && (
        <div className="mt-2 flex flex-col gap-2">
          {sources.map((source) => (
            <SourceItem key={source.chunk_id} source={source} />
          ))}
        </div>
      )}
    </div>
  );
}
