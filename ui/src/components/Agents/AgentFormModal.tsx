import { Modal } from '../common/Modal';
import { AgentForm } from './AgentForm';
import type { AgentCreate, AgentResponse, AgentUpdate } from '../../types/api';

interface AgentFormModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: AgentCreate | AgentUpdate) => void;
  agent?: AgentResponse | null;
  isLoading?: boolean;
}

export function AgentFormModal({
  isOpen,
  onClose,
  onSubmit,
  agent,
  isLoading = false,
}: AgentFormModalProps) {
  const title = agent ? 'Edit Agent' : 'Create Agent';

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={title} size="lg">
      <AgentForm
        agent={agent}
        onSubmit={onSubmit}
        onCancel={onClose}
        isLoading={isLoading}
      />
    </Modal>
  );
}
