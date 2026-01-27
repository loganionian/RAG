import { useState, useCallback } from 'react';
import type {
  AgentResponse,
  AgentsListResponse,
  AgentCreate,
  AgentUpdate,
  AgentDeleteResponse,
  ErrorResponse,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export interface UseAgentsState {
  agents: AgentResponse[];
  isLoading: boolean;
  error: string | null;
}

export interface UseAgentsResult extends UseAgentsState {
  fetchAgents: () => Promise<AgentResponse[] | null>;
  createAgent: (data: AgentCreate) => Promise<AgentResponse | null>;
  updateAgent: (id: string, data: AgentUpdate) => Promise<AgentResponse | null>;
  deleteAgent: (id: string) => Promise<boolean>;
  clearError: () => void;
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;

    try {
      const errorData = (await response.json()) as ErrorResponse;
      errorMessage = errorData.detail || errorMessage;
    } catch {
      // Use default error message if JSON parsing fails
    }

    throw new Error(errorMessage);
  }

  return response.json() as Promise<T>;
}

export function useAgents(): UseAgentsResult {
  const [state, setState] = useState<UseAgentsState>({
    agents: [],
    isLoading: false,
    error: null,
  });

  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  const fetchAgents = useCallback(async (): Promise<AgentResponse[] | null> => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const response = await fetch(`${API_BASE_URL}/api/agents`, {
        method: 'GET',
        headers: {
          Accept: 'application/json',
        },
      });

      const data = await handleResponse<AgentsListResponse>(response);
      setState({ agents: data.agents, isLoading: false, error: null });
      return data.agents;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setState((prev) => ({ ...prev, isLoading: false, error: errorMessage }));
      return null;
    }
  }, []);

  const createAgent = useCallback(async (data: AgentCreate): Promise<AgentResponse | null> => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const response = await fetch(`${API_BASE_URL}/api/agents`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(data),
      });

      const agent = await handleResponse<AgentResponse>(response);
      setState((prev) => ({
        agents: [...prev.agents, agent],
        isLoading: false,
        error: null,
      }));
      return agent;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setState((prev) => ({ ...prev, isLoading: false, error: errorMessage }));
      return null;
    }
  }, []);

  const updateAgent = useCallback(
    async (id: string, data: AgentUpdate): Promise<AgentResponse | null> => {
      setState((prev) => ({ ...prev, isLoading: true, error: null }));

      try {
        const response = await fetch(`${API_BASE_URL}/api/agents/${id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
          },
          body: JSON.stringify(data),
        });

        const agent = await handleResponse<AgentResponse>(response);
        setState((prev) => ({
          agents: prev.agents.map((a) => (a.id === id ? agent : a)),
          isLoading: false,
          error: null,
        }));
        return agent;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        setState((prev) => ({ ...prev, isLoading: false, error: errorMessage }));
        return null;
      }
    },
    []
  );

  const deleteAgent = useCallback(async (id: string): Promise<boolean> => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const response = await fetch(`${API_BASE_URL}/api/agents/${id}`, {
        method: 'DELETE',
        headers: {
          Accept: 'application/json',
        },
      });

      await handleResponse<AgentDeleteResponse>(response);
      setState((prev) => ({
        agents: prev.agents.filter((a) => a.id !== id),
        isLoading: false,
        error: null,
      }));
      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setState((prev) => ({ ...prev, isLoading: false, error: errorMessage }));
      return false;
    }
  }, []);

  return {
    ...state,
    fetchAgents,
    createAgent,
    updateAgent,
    deleteAgent,
    clearError,
  };
}
