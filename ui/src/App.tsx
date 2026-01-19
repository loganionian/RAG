import { Routes, Route } from 'react-router-dom';
import { useEffect, useCallback, useState } from 'react';
import { Header, Sidebar, Container } from './components/Layout';
import type { HealthResponse } from './types/api';

function ChatPlaceholder() {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center">
      <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">Chat</h2>
      <p className="mt-2 text-gray-600 dark:text-gray-400">
        Chat interface will be implemented in issue #107
      </p>
    </div>
  );
}

function DocumentsPlaceholder() {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center">
      <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">Documents</h2>
      <p className="mt-2 text-gray-600 dark:text-gray-400">
        Document management will be implemented in issue #108
      </p>
    </div>
  );
}

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isChecking, setIsChecking] = useState(true);

  const checkHealth = useCallback(async () => {
    setIsChecking(true);
    try {
      const response = await fetch('/api/health');
      if (response.ok) {
        const data = (await response.json()) as HealthResponse;
        setIsConnected(data.status === 'healthy');
      } else {
        setIsConnected(false);
      }
    } catch {
      setIsConnected(false);
    } finally {
      setIsChecking(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Header isConnected={isConnected} isChecking={isChecking} />
      <Sidebar />
      <Container>
        <Routes>
          <Route path="/" element={<ChatPlaceholder />} />
          <Route path="/documents" element={<DocumentsPlaceholder />} />
        </Routes>
      </Container>
    </div>
  );
}

export default App;
