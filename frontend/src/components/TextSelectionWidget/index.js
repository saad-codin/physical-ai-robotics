import React, { useState, useEffect, useRef } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

function TextSelectionWidgetComponent() {
  const [selectedText, setSelectedText] = useState('');
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isVisible, setIsVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState('');
  const [showResponse, setShowResponse] = useState(false);
  const widgetRef = useRef(null);

  // Function to handle text selection
  const handleSelection = () => {
    const selection = window.getSelection();
    const text = selection.toString().trim();

    if (text.length > 0 && text.length < 500) { // Limit selection length
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();

      setPosition({
        x: rect.left + window.scrollX,
        y: rect.top + window.scrollY - 40 // Position above the selection
      });

      setSelectedText(text);
      setIsVisible(true);
      setResponse('');
      setShowResponse(false);
    } else {
      setIsVisible(false);
      setResponse('');
      setShowResponse(false);
    }
  };

  // Function to handle asking AI
  const handleAskAI = async () => {
    if (!selectedText) return;

    setIsLoading(true);
    setResponse('');
    setShowResponse(true);

    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch('http://localhost:8000/v1/chatkit/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` })
        },
        body: JSON.stringify({
          message: `Explain this concept: "${selectedText}" in the context of robotics and AI.`,
          query: `Explain this concept: "${selectedText}" in the context of robotics and AI.`
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();
      setResponse(data.event?.text || 'Received response from AI tutor.');
    } catch (error) {
      console.error('Error calling AI tutor:', error);
      setResponse('Sorry, I encountered an error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Close widget when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (widgetRef.current && !widgetRef.current.contains(event.target)) {
        setIsVisible(false);
        setResponse('');
        setShowResponse(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Listen for text selection
  useEffect(() => {
    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  if (!isVisible) {
    return null;
  }

  return (
    <div
      ref={widgetRef}
      style={{
        position: 'absolute',
        left: `${position.x}px`,
        top: `${position.y}px`,
        zIndex: 10000,
        backgroundColor: '#4299e1',
        color: 'white',
        padding: '8px 12px',
        borderRadius: '6px',
        fontSize: '14px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        transform: 'translateY(-100%)',
        minWidth: '120px',
        maxWidth: '300px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <button
          onClick={handleAskAI}
          disabled={isLoading}
          style={{
            background: 'none',
            border: 'none',
            color: 'white',
            fontSize: '12px',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            padding: '4px 8px',
            borderRadius: '4px',
            backgroundColor: 'rgba(255, 255, 255, 0.2)',
          }}
        >
          {isLoading ? 'Asking...' : '🤖 Ask AI'}
        </button>
        <button
          onClick={() => {
            setIsVisible(false);
            setResponse('');
            setShowResponse(false);
          }}
          style={{
            background: 'none',
            border: 'none',
            color: 'white',
            fontSize: '16px',
            cursor: 'pointer',
            padding: '0',
            width: '20px',
            height: '20px',
          }}
        >
          ×
        </button>
      </div>

      {showResponse && response && (
        <div
          style={{
            marginTop: '8px',
            paddingTop: '8px',
            borderTop: '1px solid rgba(255, 255, 255, 0.3)',
            fontSize: '12px',
            lineHeight: '1.4',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '4px',
            padding: '6px',
          }}
        >
          {response}
        </div>
      )}
    </div>
  );
}

function TextSelectionWidget() {
  return (
    <BrowserOnly>
      {() => <TextSelectionWidgetComponent />}
    </BrowserOnly>
  );
}

export default TextSelectionWidget;