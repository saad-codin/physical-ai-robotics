import React, { useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

function ChatKitComponent() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sampleMessages = [
    { id: 1, type: 'assistant', content: 'Hello! I\'m your AI Robotics Tutor. Ask me anything about robotics concepts!' },
  ];

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage = { id: Date.now(), type: 'user', content: inputValue };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Get auth token if user is logged in
      const token = localStorage.getItem('access_token');

      // Call the backend ChatKit endpoint
      const response = await fetch('http://localhost:8000/v1/chatkit/chatkit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` })
        },
        body: JSON.stringify({ message: inputValue })
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();
      const botResponse = {
        id: Date.now() + 1,
        type: 'assistant',
        content: data.event?.text || 'I received your question and am processing it.'
      };
      setMessages(prev => [...prev, botResponse]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage();
  };

  return (
    <BrowserOnly>
      {() => (
        <div className="chatkit-container" style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          zIndex: 1000
        }}>
          {!isOpen ? (
            <button
              onClick={() => setIsOpen(true)}
              style={{
                backgroundColor: '#4299e1',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                width: '60px',
                height: '60px',
                fontSize: '24px',
                cursor: 'pointer',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
              }}
            >
              🤖
            </button>
          ) : (
            <div style={{
              width: '400px',
              height: '500px',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              backgroundColor: 'white',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: '0 10px 15px rgba(0, 0, 0, 0.1)',
              overflow: 'hidden'
            }}>
              <div style={{
                padding: '1rem',
                backgroundColor: '#4299e1',
                color: 'white',
                borderTopLeftRadius: '8px',
                borderTopRightRadius: '8px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <span>AI Robotics Tutor</span>
                <button
                  onClick={() => setIsOpen(false)}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'white',
                    fontSize: '1.2rem',
                    cursor: 'pointer'
                  }}
                >
                  ×
                </button>
              </div>

              <div style={{
                flex: 1,
                padding: '1rem',
                overflowY: 'auto',
                backgroundColor: '#f8fafc',
                display: 'flex',
                flexDirection: 'column'
              }}>
                {(messages.length === 0 ? sampleMessages : messages).map((message) => (
                  <div
                    key={message.id}
                    style={{
                      marginBottom: '1rem',
                      textAlign: message.type === 'user' ? 'right' : 'left'
                    }}
                  >
                    <div
                      style={{
                        display: 'inline-block',
                        padding: '0.5rem 1rem',
                        borderRadius: '8px',
                        backgroundColor: message.type === 'user' ? '#4299e1' : '#ffffff',
                        color: message.type === 'user' ? 'white' : '#333',
                        border: message.type === 'assistant' ? '1px solid #e2e8f0' : 'none',
                        maxWidth: '80%'
                      }}
                    >
                      {message.content}
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div style={{ textAlign: 'left', marginBottom: '1rem' }}>
                    <div style={{
                      display: 'inline-block',
                      padding: '0.5rem 1rem',
                      borderRadius: '8px',
                      backgroundColor: '#ffffff',
                      color: '#333',
                      border: '1px solid #e2e8f0'
                    }}>
                      Thinking...
                    </div>
                  </div>
                )}
              </div>

              <form onSubmit={handleSubmit} style={{ padding: '1rem', borderTop: '1px solid #e2e8f0' }}>
                <div style={{ display: 'flex' }}>
                  <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="Ask about robotics concepts..."
                    style={{
                      flex: 1,
                      padding: '0.5rem',
                      border: '1px solid #e2e8f0',
                      borderRadius: '4px 0 0 4px'
                    }}
                    disabled={isLoading}
                  />
                  <button
                    type="submit"
                    style={{
                      padding: '0.5rem 1rem',
                      backgroundColor: '#4299e1',
                      color: 'white',
                      border: 'none',
                      borderRadius: '0 4px 4px 0',
                      cursor: isLoading ? 'not-allowed' : 'pointer'
                    }}
                    disabled={isLoading}
                  >
                    Send
                  </button>
                </div>
              </form>
            </div>
          )}
        </div>
      )}
    </BrowserOnly>
  );
}

export default ChatKitComponent;