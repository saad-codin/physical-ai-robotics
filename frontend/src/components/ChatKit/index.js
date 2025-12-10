import React, { useState, useRef, useEffect } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import './styles.css';

function ChatKitComponent() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = { id: Date.now(), type: 'user', content: inputValue };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch('https://dead-stacee-galx-311dba08.koyeb.app/v1/chatkit/chat', {
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

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <BrowserOnly>
      {() => (
        <div className="chatkit-container">
          {!isOpen ? (
            <button
              className="chat-button"
              onClick={() => setIsOpen(true)}
              aria-label="Open chat"
            >
              🤖
            </button>
          ) : (
            <div className="chat-window">
              <div className="chat-header">
                <div className="chat-header-content">
                  <div className="chat-avatar">
                    🤖
                  </div>
                  <div className="chat-title">
                    <div className="chat-title-main">AI Robotics Tutor</div>
                    <div className="chat-title-status">
                      <span className="status-indicator"></span>
                      Online
                    </div>
                  </div>
                </div>
                <button
                  className="close-button"
                  onClick={() => setIsOpen(false)}
                  aria-label="Close chat"
                >
                  ×
                </button>
              </div>

              <div className="chat-messages">
                {messages.length === 0 && (
                  <div className="message assistant">
                    <div className="message-avatar">🤖</div>
                    <div className="message-content">
                      Hello! I'm your AI Robotics Tutor. Ask me anything about robotics concepts, ROS 2, AI, or VLA!
                    </div>
                  </div>
                )}

                {messages.map((message) => (
                  <div key={message.id} className={`message ${message.type}`}>
                    <div className="message-avatar">
                      {message.type === 'assistant' ? '🤖' : '👤'}
                    </div>
                    <div className="message-content">
                      {message.content}
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="message assistant">
                    <div className="message-avatar">🤖</div>
                    <div className="loading-indicator">
                      <div className="loading-dots">
                        <div className="loading-dot"></div>
                        <div className="loading-dot"></div>
                        <div className="loading-dot"></div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              <div className="chat-input-container">
                <form onSubmit={handleSubmit} className="chat-input-wrapper">
                  <textarea
                    className="chat-input"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask about robotics concepts..."
                    disabled={isLoading}
                    rows={1}
                  />
                  <button
                    type="submit"
                    className="send-button"
                    disabled={isLoading || !inputValue.trim()}
                    aria-label="Send message"
                  >
                    ➤
                  </button>
                </form>
              </div>
            </div>
          )}
        </div>
      )}
    </BrowserOnly>
  );
}

export default ChatKitComponent;
