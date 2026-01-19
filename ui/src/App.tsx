import { Routes, Route } from 'react-router-dom';
import { useEffect, useCallback, useState } from 'react';
import { Header, Sidebar, Container } from './components/Layout';
import { Chat } from './components/Chat';
import { Documents } from './components/Documents';
import type { HealthResponse } from './types/api';

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
          <Route path="/" element={<Chat />} />
          <Route path="/documents" element={<Documents />} />
        </Routes>
      </Container>
    </div>
  );
}

export default App;
