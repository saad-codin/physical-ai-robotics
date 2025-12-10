import React from 'react';
import { AuthProvider } from '../context/AuthContext';
import { ToastProvider } from '../context/ToastContext';
import ChatKitComponent from '../components/ChatKit';
import TextSelectionWidget from '../components/TextSelectionWidget';
import ReadingProgress from '../components/ReadingProgress';

// Root component wrapper for Docusaurus
// This wraps the entire app and provides global context
export default function Root({ children }) {
  return (
    <ToastProvider>
      <AuthProvider>
        <ReadingProgress />
        {children}
        <ChatKitComponent />
        <TextSelectionWidget />
      </AuthProvider>
    </ToastProvider>
  );
}