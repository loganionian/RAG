import { Check, X, RefreshCw } from 'lucide-react';

interface UploadProgressProps {
  filename: string;
  progress: number;
  status: 'uploading' | 'success' | 'error';
  chunksCreated?: number;
  error?: string;
  onRetry?: () => void;
}

export function UploadProgress({
  filename,
  progress,
  status,
  chunksCreated,
  error,
  onRetry,
}: UploadProgressProps) {
  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <span className="truncate font-medium text-gray-900 dark:text-gray-100">{filename}</span>
        {status === 'success' && <Check className="h-5 w-5 text-green-500" />}
        {status === 'error' && <X className="h-5 w-5 text-red-500" />}
      </div>

      {status === 'uploading' && (
        <div className="mt-2">
          <div className="h-2 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
            <div
              className="h-full bg-primary-600 transition-all duration-300 dark:bg-primary-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Uploading... {progress}%</p>
        </div>
      )}

      {status === 'success' && chunksCreated !== undefined && (
        <p className="mt-2 text-sm text-green-600 dark:text-green-400">
          {chunksCreated} chunks created
        </p>
      )}

      {status === 'error' && (
        <div className="mt-2">
          <p className="text-sm text-red-600 dark:text-red-400">{error || 'Upload failed'}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-2 inline-flex items-center gap-1 text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300"
            >
              <RefreshCw className="h-4 w-4" />
              Retry
            </button>
          )}
        </div>
      )}
    </div>
  );
}
