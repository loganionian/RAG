import { AlertTriangle } from 'lucide-react';
import { Modal } from './Modal';

interface DeleteConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  itemName: string;
  itemType?: string;
  isLoading?: boolean;
}

export function DeleteConfirmModal({
  isOpen,
  onClose,
  onConfirm,
  itemName,
  itemType = 'item',
  isLoading = false,
}: DeleteConfirmModalProps) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`Delete ${itemType}`} size="sm">
      <div className="flex flex-col items-center text-center">
        <div className="mb-4 rounded-full bg-red-100 p-3 dark:bg-red-900/30">
          <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
        </div>
        <p className="mb-2 text-gray-700 dark:text-gray-300">
          Are you sure you want to delete this {itemType}?
        </p>
        <p className="mb-6 font-medium text-gray-900 dark:text-gray-100">"{itemName}"</p>
        <p className="mb-6 text-sm text-gray-500 dark:text-gray-400">
          This action cannot be undone.
        </p>
        <div className="flex w-full gap-3">
          <button
            type="button"
            onClick={onClose}
            disabled={isLoading}
            className="btn btn-secondary flex-1"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isLoading}
            className="btn flex-1 bg-red-600 text-white hover:bg-red-700 disabled:opacity-50"
          >
            {isLoading ? 'Deleting...' : 'Delete'}
          </button>
        </div>
      </div>
    </Modal>
  );
}
