import { FileText, FileSpreadsheet, File } from 'lucide-react';
import type { DocumentInfo } from '../../types/api';

interface DocumentRowProps {
  document: DocumentInfo;
}

function getFileIcon(fileType: string) {
  switch (fileType.toLowerCase()) {
    case 'pdf':
    case 'doc':
    case 'docx':
    case 'md':
    case 'txt':
      return FileText;
    case 'xlsx':
    case 'xls':
    case 'csv':
    case 'tsv':
      return FileSpreadsheet;
    default:
      return File;
  }
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

export function DocumentRow({ document }: DocumentRowProps) {
  const Icon = getFileIcon(document.file_type);

  return (
    <tr className="border-b border-gray-200 dark:border-gray-700">
      <td className="py-3 pr-4">
        <div className="flex items-center gap-3">
          <Icon className="h-5 w-5 flex-shrink-0 text-gray-400" />
          <span className="truncate font-medium text-gray-900 dark:text-gray-100">
            {document.filename}
          </span>
        </div>
      </td>
      <td className="py-3 pr-4">
        <span className="rounded bg-gray-100 px-2 py-1 text-xs font-medium uppercase text-gray-600 dark:bg-gray-700 dark:text-gray-300">
          {document.file_type}
        </span>
      </td>
      <td className="py-3 pr-4">
        <span className="text-gray-600 dark:text-gray-400">{document.chunks}</span>
      </td>
      <td className="py-3">
        <span className="text-gray-500 dark:text-gray-400">
          {formatDate(document.ingested_at)}
        </span>
      </td>
    </tr>
  );
}
