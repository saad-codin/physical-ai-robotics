import React, { useState, useEffect, useRef } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import './styles.css';

function TextSelectionWidgetComponent() {
  const [selectedText, setSelectedText] = useState('');
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isVisible, setIsVisible] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [additionalContext, setAdditionalContext] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState('');
  const widgetRef = useRef(null);
  const modalRef = useRef(null);

  // Function to handle text selection
  const handleSelection = () => {
    const selection = window.getSelection();
    const text = selection.toString().trim();

    if (text.length > 0 && text.length < 1000) { // Limit selection length
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();

      setPosition({
        x: rect.left + window.scrollX + (rect.width / 2),
        y: rect.top + window.scrollY - 50 // Position above the selection
      });

      setSelectedText(text);
      setIsVisible(true);
      setResponse('');
      setAdditionalContext('');
    } else if (text.length === 0) {
      setIsVisible(false);
    }
  };

  // Open the modal for adding context
  const handleOpenModal = () => {
    setShowModal(true);
    setIsVisible(false);
  };

  // Function to handle asking AI with context
  const handleAskAI = async () => {
    if (!selectedText) return;

    setIsLoading(true);
    setResponse('');

    try {
      // Build the query with selected text and additional context
      let query = `Explain this concept from the textbook: "${selectedText}"`;
      if (additionalContext.trim()) {
        query += `\n\nAdditional context/question: ${additionalContext}`;
      }
      query += '\n\nPlease provide a clear, concise explanation in the context of robotics and AI.';

      const token = localStorage.getItem('access_token');
      const response = await fetch('https://dead-stacee-galx-311dba08.koyeb.app/v1/chatkit/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` })
        },
        body: JSON.stringify({
          message: query,
          query: query
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();
      setResponse(data.event?.text || 'Received response from AI tutor.');
    } catch (error) {
      console.error('Error calling AI tutor:', error);
      setResponse('Sorry, I encountered an error. Please try again. Make sure the backend server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  // Close modal
  const handleCloseModal = () => {
    setShowModal(false);
    setAdditionalContext('');
    setResponse('');
  };

  // Close widget when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (widgetRef.current && !widgetRef.current.contains(event.target)) {
        setIsVisible(false);
      }
      if (showModal && modalRef.current && !modalRef.current.contains(event.target)) {
        handleCloseModal();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showModal]);

  // Listen for text selection
  useEffect(() => {
    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  // Handle Escape key
  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        setIsVisible(false);
        handleCloseModal();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, []);

  return (
    <>
      {/* Text Selection Widget */}
      {isVisible && (
        <div
          ref={widgetRef}
          className="text-selection-widget"
          style={{
            left: `${position.x}px`,
            top: `${position.y}px`,
          }}
        >
          <button
            onClick={handleOpenModal}
            className="ask-ai-button"
          >
            <span className="button-icon">🤖</span>
            <span className="button-text">Ask AI</span>
          </button>
        </div>
      )}

      {/* AI Chat Modal */}
      {showModal && (
        <div className="modal-overlay">
          <div ref={modalRef} className="modal-container">
            {/* Modal Header */}
            <div className="modal-header">
              <h3 className="modal-title">
                <span className="modal-icon">🤖</span>
                Ask AI About This
              </h3>
              <button
                onClick={handleCloseModal}
                className="close-button"
                aria-label="Close"
              >
                ×
              </button>
            </div>

            {/* Selected Text Display */}
            <div className="selected-text-display">
              <div className="selected-text-label">Selected Text:</div>
              <div className="selected-text-content">
                "{selectedText}"
              </div>
            </div>

            {/* Additional Context Input */}
            <div className="context-input-section">
              <label htmlFor="context-input" className="context-label">
                Add more context or ask a specific question (optional):
              </label>
              <textarea
                id="context-input"
                className="context-textarea"
                placeholder="e.g., How does this relate to ROS 2? Can you give me an example? What are the practical applications?"
                value={additionalContext}
                onChange={(e) => setAdditionalContext(e.target.value)}
                rows={4}
                disabled={isLoading}
              />
            </div>

            {/* Action Buttons */}
            <div className="modal-actions">
              <button
                onClick={handleAskAI}
                disabled={isLoading}
                className="submit-button"
              >
                {isLoading ? (
                  <>
                    <span className="spinner"></span>
                    <span>Asking AI...</span>
                  </>
                ) : (
                  <>
                    <span>✨</span>
                    <span>Ask AI</span>
                  </>
                )}
              </button>
              <button
                onClick={handleCloseModal}
                className="cancel-button"
                disabled={isLoading}
              >
                Cancel
              </button>
            </div>

            {/* Response Display */}
            {response && (
              <div className="response-section">
                <div className="response-header">
                  <span className="response-icon">💡</span>
                  <span className="response-title">AI Response:</span>
                </div>
                <div className="response-content">
                  {response}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
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