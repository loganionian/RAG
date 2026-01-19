import { ReactNode } from 'react';

interface ContainerProps {
  children: ReactNode;
}

export function Container({ children }: ContainerProps) {
  return (
    <main className="ml-0 mt-14 min-h-[calc(100vh-3.5rem)] p-4 md:ml-56 md:p-6">
      {children}
    </main>
  );
}
