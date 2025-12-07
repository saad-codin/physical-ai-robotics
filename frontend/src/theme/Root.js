import React from 'react';
import { AuthProvider } from '../context/AuthContext';
import { ToastProvider } from '../context/ToastContext';
import ChatKitComponent from '../components/ChatKit';

// Root component wrapper for Docusaurus
// This wraps the entire app and provides global context
export default function Root({ children }) {
  return (
    <ToastProvider>
      <AuthProvider>
        {children}
        <ChatKitComponent />
      </AuthProvider>
    </ToastProvider>
  );
}