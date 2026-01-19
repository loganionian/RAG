import { useState, useCallback } from 'react';
import type { ErrorResponse } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export interface ApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export interface ApiResult<T> extends ApiState<T> {
  execute: (...args: unknown[]) => Promise<T | null>;
  reset: () => void;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public errorType?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
    let errorType: string | undefined;

    try {
      const errorData = (await response.json()) as ErrorResponse;
      errorMessage = errorData.detail || errorMessage;
      errorType = errorData.error_type;
    } catch {
      // Use default error message if JSON parsing fails
    }

    throw new ApiError(errorMessage, response.status, errorType);
  }

  return response.json() as Promise<T>;
}

export function useApi<T>(): ApiResult<T> & {
  get: (path: string) => Promise<T | null>;
  post: (path: string, body: unknown) => Promise<T | null>;
  postFormData: (path: string, formData: FormData) => Promise<T | null>;
} {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null });
  }, []);

  const get = useCallback(async (path: string): Promise<T | null> => {
    setState({ data: null, loading: true, error: null });

    try {
      const response = await fetch(`${API_BASE_URL}${path}`, {
        method: 'GET',
        headers: {
          Accept: 'application/json',
        },
      });

      const data = await handleResponse<T>(response);
      setState({ data, loading: false, error: null });
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setState({ data: null, loading: false, error: errorMessage });
      return null;
    }
  }, []);

  const post = useCallback(async (path: string, body: unknown): Promise<T | null> => {
    setState({ data: null, loading: true, error: null });

    try {
      const response = await fetch(`${API_BASE_URL}${path}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(body),
      });

      const data = await handleResponse<T>(response);
      setState({ data, loading: false, error: null });
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setState({ data: null, loading: false, error: errorMessage });
      return null;
    }
  }, []);

  const postFormData = useCallback(async (path: string, formData: FormData): Promise<T | null> => {
    setState({ data: null, loading: true, error: null });

    try {
      const response = await fetch(`${API_BASE_URL}${path}`, {
        method: 'POST',
        headers: {
          Accept: 'application/json',
        },
        body: formData,
      });

      const data = await handleResponse<T>(response);
      setState({ data, loading: false, error: null });
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setState({ data: null, loading: false, error: errorMessage });
      return null;
    }
  }, []);

  const execute = useCallback(async (): Promise<T | null> => {
    // Generic execute function - can be extended if needed
    return null;
  }, []);

  return {
    ...state,
    execute,
    reset,
    get,
    post,
    postFormData,
  };
}

/** Hook for health check polling */
export function useHealthCheck() {
  const api = useApi<{ status: string; vectorstore: { healthy: boolean }; llm_provider: { healthy: boolean } }>();

  const checkHealth = useCallback(async () => {
    return api.get('/api/health');
  }, [api]);

  return {
    ...api,
    checkHealth,
    isConnected: api.data?.status === 'healthy',
  };
}
