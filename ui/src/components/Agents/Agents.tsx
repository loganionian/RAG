import { useState, useEffect, useCallback } from 'react';
import { useAgents } from '../../hooks/useAgents';
import type { AgentCreate, AgentResponse, AgentUpdate } from '../../types/api';
import { AgentList } from './AgentList';
import { AgentFormModal } from './AgentFormModal';
import { DeleteConfirmModal } from '../common/DeleteConfirmModal';

export function Agents() {
  const { agents, isLoading, error, fetchAgents, createAgent, updateAgent, deleteAgent, clearError } =
    useAgents();

  const [isFormModalOpen, setIsFormModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<AgentResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    fetchAgents();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleCreate = useCallback(() => {
    setSelectedAgent(null);
    setIsFormModalOpen(true);
  }, []);

  const handleEdit = useCallback((agent: AgentResponse) => {
    setSelectedAgent(agent);
    setIsFormModalOpen(true);
  }, []);

  const handleDeleteClick = useCallback((agent: AgentResponse) => {
    setSelectedAgent(agent);
    setIsDeleteModalOpen(true);
  }, []);

  const handleFormSubmit = useCallback(
    async (data: AgentCreate | AgentUpdate) => {
      setIsSubmitting(true);
      clearError();

      let result;
      if (selectedAgent) {
        result = await updateAgent(selectedAgent.id, data as AgentUpdate);
      } else {
        result = await createAgent(data as AgentCreate);
      }

      setIsSubmitting(false);

      if (result) {
        setIsFormModalOpen(false);
        setSelectedAgent(null);
      }
    },
    [selectedAgent, createAgent, updateAgent, clearError]
  );

  const handleDeleteConfirm = useCallback(async () => {
    if (!selectedAgent) return;

    setIsSubmitting(true);
    clearError();

    const success = await deleteAgent(selectedAgent.id);

    setIsSubmitting(false);

    if (success) {
      setIsDeleteModalOpen(false);
      setSelectedAgent(null);
    }
  }, [selectedAgent, deleteAgent, clearError]);

  const handleCloseFormModal = useCallback(() => {
    setIsFormModalOpen(false);
    setSelectedAgent(null);
    clearError();
  }, [clearError]);

  const handleCloseDeleteModal = useCallback(() => {
    setIsDeleteModalOpen(false);
    setSelectedAgent(null);
    clearError();
  }, [clearError]);

  return (
    <div className="flex flex-col gap-6 p-4">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">Agents</h1>
        <p className="mt-1 text-gray-600 dark:text-gray-400">
          Configure and manage LLM agent personalities
        </p>
      </div>

      {error && (
        <div className="rounded-lg bg-red-50 p-4 text-red-700 dark:bg-red-900/30 dark:text-red-400">
          {error}
        </div>
      )}

      <AgentList
        agents={agents}
        isLoading={isLoading}
        onRefresh={fetchAgents}
        onCreate={handleCreate}
        onEdit={handleEdit}
        onDelete={handleDeleteClick}
      />

      <AgentFormModal
        isOpen={isFormModalOpen}
        onClose={handleCloseFormModal}
        onSubmit={handleFormSubmit}
        agent={selectedAgent}
        isLoading={isSubmitting}
      />

      <DeleteConfirmModal
        isOpen={isDeleteModalOpen}
        onClose={handleCloseDeleteModal}
        onConfirm={handleDeleteConfirm}
        itemName={selectedAgent?.name || ''}
        itemType="agent"
        isLoading={isSubmitting}
      />
    </div>
  );
}
