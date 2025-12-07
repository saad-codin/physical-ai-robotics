import React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

function RoboticsCodeBlock({ children, language, title }) {
  return (
    <BrowserOnly>
      {() => {
        const [copied, setCopied] = React.useState(false);

        const copyToClipboard = () => {
          navigator.clipboard.writeText(children);
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        };

        return (
          <div className="robotics-card">
            {title && <h4>{title}</h4>}
            <div className="code-block-container">
              <pre className="ros-code-block">
                <code className={`language-${language}`}>
                  {children}
                </code>
              </pre>
            </div>
            <div className="button-container" style={{ marginTop: '10px', textAlign: 'right' }}>
              <button
                className="interactive-btn"
                onClick={copyToClipboard}
                style={{
                  backgroundColor: copied ? '#48bb78' : '#4299e1',
                  marginRight: '10px'
                }}
              >
                {copied ? '✓ Copied!' : 'Copy Code'}
              </button>
              <button className="interactive-btn" style={{ backgroundColor: '#48bb78' }}>
                Try in Simulator
              </button>
            </div>
          </div>
        );
      }}
    </BrowserOnly>
  );
}

export default RoboticsCodeBlock;