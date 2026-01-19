import { useState, useEffect, useCallback } from 'react';
import { useApi } from '../../hooks/useApi';
import type { DocumentsResponse, IngestResponse } from '../../types/api';
import { UploadZone } from './UploadZone';
import { UploadProgress } from './UploadProgress';
import { DocumentList } from './DocumentList';

interface UploadState {
  file: File;
  progress: number;
  status: 'uploading' | 'success' | 'error';
  chunksCreated?: number;
  error?: string;
}

export function Documents() {
  const [uploads, setUploads] = useState<UploadState[]>([]);
  const documentsApi = useApi<DocumentsResponse>();
  const uploadApi = useApi<IngestResponse>();

  const fetchDocuments = useCallback(async () => {
    await documentsApi.get('/api/documents');
  }, [documentsApi]);

  useEffect(() => {
    fetchDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleUpload = useCallback(
    async (file: File) => {
      const uploadIndex = uploads.length;
      setUploads((prev) => [...prev, { file, progress: 0, status: 'uploading' }]);

      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploads((prev) =>
          prev.map((u, i) =>
            i === uploadIndex && u.status === 'uploading' && u.progress < 90
              ? { ...u, progress: u.progress + 10 }
              : u
          )
        );
      }, 200);

      const formData = new FormData();
      formData.append('file', file);

      const response = await uploadApi.postFormData('/api/ingest', formData);

      clearInterval(progressInterval);

      if (response) {
        setUploads((prev) =>
          prev.map((u, i) =>
            i === uploadIndex
              ? { ...u, progress: 100, status: 'success', chunksCreated: response.chunks_created }
              : u
          )
        );
        // Refresh document list
        fetchDocuments();
      } else {
        setUploads((prev) =>
          prev.map((u, i) =>
            i === uploadIndex
              ? { ...u, status: 'error', error: uploadApi.error || 'Upload failed' }
              : u
          )
        );
      }
    },
    [uploads.length, uploadApi, fetchDocuments]
  );

  const handleRetry = (index: number) => {
    const upload = uploads[index];
    if (upload) {
      // Remove failed upload
      setUploads((prev) => prev.filter((_, i) => i !== index));
      // Retry upload
      handleUpload(upload.file);
    }
  };

  const isUploading = uploads.some((u) => u.status === 'uploading');

  return (
    <div className="flex flex-col gap-6 p-4">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">Documents</h1>
        <p className="mt-1 text-gray-600 dark:text-gray-400">
          Upload and manage your documents for RAG retrieval
        </p>
      </div>

      <UploadZone onUpload={handleUpload} isUploading={isUploading} />

      {uploads.length > 0 && (
        <div className="flex flex-col gap-3">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Recent Uploads</h3>
          {uploads.map((upload, index) => (
            <UploadProgress
              key={`${upload.file.name}-${index}`}
              filename={upload.file.name}
              progress={upload.progress}
              status={upload.status}
              chunksCreated={upload.chunksCreated}
              error={upload.error}
              onRetry={upload.status === 'error' ? () => handleRetry(index) : undefined}
            />
          ))}
        </div>
      )}

      <DocumentList
        documents={documentsApi.data?.documents || []}
        isLoading={documentsApi.loading}
        onRefresh={fetchDocuments}
      />
    </div>
  );
}
