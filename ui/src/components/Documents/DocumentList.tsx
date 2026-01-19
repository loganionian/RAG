import { RefreshCw, FileText } from 'lucide-react';
import type { DocumentInfo } from '../../types/api';
import { DocumentRow } from './DocumentRow';

interface DocumentListProps {
  documents: DocumentInfo[];
  isLoading: boolean;
  onRefresh: () => void;
}

export function DocumentList({ documents, isLoading, onRefresh }: DocumentListProps) {
  if (documents.length === 0 && !isLoading) {
    return (
      <div className="card flex flex-col items-center justify-center py-12">
        <FileText className="h-12 w-12 text-gray-300 dark:text-gray-600" />
        <p className="mt-3 text-gray-600 dark:text-gray-400">No documents ingested yet</p>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-500">
          Upload a document to get started
        </p>
      </div>
    );
  }

  return (
    <div className="card overflow-hidden p-0">
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3 dark:border-gray-700">
        <h3 className="font-medium text-gray-900 dark:text-gray-100">
          Documents ({documents.length})
        </h3>
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="btn btn-secondary p-2"
          aria-label="Refresh documents"
        >
          <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-200 bg-gray-50 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:border-gray-700 dark:bg-gray-800/50 dark:text-gray-400">
              <th className="px-4 py-3">Name</th>
              <th className="px-4 py-3">Type</th>
              <th className="px-4 py-3">Chunks</th>
              <th className="px-4 py-3">Ingested</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {documents.map((doc) => (
              <DocumentRow key={doc.doc_id} document={doc} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
