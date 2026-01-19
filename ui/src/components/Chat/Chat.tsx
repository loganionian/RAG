import { useState, useCallback } from 'react';
import { MessageSquare } from 'lucide-react';
import { useApi } from '../../hooks/useApi';
import type { QueryResponse } from '../../types/api';
import { MessageList, type ChatMessage } from './MessageList';
import { ChatInput } from './ChatInput';

const EXAMPLE_QUESTIONS = [
  'What are the key findings in the documents?',
  'Summarize the main topics covered.',
  'What skills are mentioned as important?',
];

export function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const api = useApi<QueryResponse>();

  const handleSubmit = useCallback(
    async (question: string) => {
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: question,
      };
      setMessages((prev) => [...prev, userMessage]);

      const response = await api.post('/api/query', { question, k: 5 });

      if (response) {
        const assistantMessage: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else if (api.error) {
        const errorMessage: ChatMessage = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `Sorry, I encountered an error: ${api.error}`,
        };
        setMessages((prev) => [...prev, errorMessage]);
      }
    },
    [api]
  );

  const handleExampleClick = (question: string) => {
    handleSubmit(question);
  };

  if (messages.length === 0) {
    return (
      <div className="flex h-full flex-col">
        <div className="flex flex-1 flex-col items-center justify-center p-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary-100 dark:bg-primary-900/30">
            <MessageSquare className="h-8 w-8 text-primary-600 dark:text-primary-400" />
          </div>
          <h2 className="mt-4 text-xl font-semibold text-gray-900 dark:text-gray-100">
            Ask a Question
          </h2>
          <p className="mt-2 max-w-md text-center text-gray-600 dark:text-gray-400">
            Chat with your documents using AI-powered retrieval. Ask questions and get answers
            based on the content of your uploaded files.
          </p>
          <div className="mt-6 flex flex-col gap-2">
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Try an example:
            </p>
            {EXAMPLE_QUESTIONS.map((question) => (
              <button
                key={question}
                onClick={() => handleExampleClick(question)}
                className="rounded-lg border border-gray-200 bg-white px-4 py-2 text-left text-sm text-gray-700 transition-colors hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
        <ChatInput onSubmit={handleSubmit} isLoading={api.loading} />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <MessageList messages={messages} isLoading={api.loading} />
      <ChatInput onSubmit={handleSubmit} isLoading={api.loading} />
    </div>
  );
}
