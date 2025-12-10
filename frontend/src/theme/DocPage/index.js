import React, { useEffect } from 'react';
import DocPage from '@theme-original/DocPage';
import { useAuth } from '../../context/AuthContext';
import { useHistory } from '@docusaurus/router';
import BrowserOnly from '@docusaurus/BrowserOnly';

function DocPageWithAuth(props) {
  return (
    <BrowserOnly fallback={<div>Loading...</div>}>
      {() => <DocPageContent {...props} />}
    </BrowserOnly>
  );
}

function DocPageContent(props) {
  const { isAuthenticated, loading } = useAuth();
  const history = useHistory();

  useEffect(() => {
    if (!loading && !isAuthenticated) {
      // Redirect to login page if not authenticated
      history.push('/login');
    }
  }, [isAuthenticated, loading, history]);

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '80vh',
        background: 'linear-gradient(135deg, #0F172A 0%, #1E293B 100%)',
      }}>
        <div style={{
          textAlign: 'center',
          color: '#F1F5F9',
        }}>
          <div style={{
            width: '60px',
            height: '60px',
            border: '4px solid rgba(66, 153, 225, 0.2)',
            borderTop: '4px solid #4299E1',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 20px',
          }} />
          <p style={{ fontSize: '18px', fontWeight: 600 }}>Loading...</p>
        </div>
        <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '80vh',
        background: 'linear-gradient(135deg, #0F172A 0%, #1E293B 100%)',
        padding: '40px 20px',
      }}>
        <div style={{
          textAlign: 'center',
          color: '#F1F5F9',
          maxWidth: '500px',
        }}>
          <div style={{
            fontSize: '72px',
            marginBottom: '20px',
          }}>🔒</div>
          <h2 style={{
            fontSize: '32px',
            fontWeight: 700,
            marginBottom: '16px',
            background: 'linear-gradient(135deg, #FFFFFF 0%, #4299E1 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Authentication Required
          </h2>
          <p style={{
            fontSize: '16px',
            color: '#94A3B8',
            marginBottom: '30px',
            lineHeight: 1.6,
          }}>
            You need to be logged in to access the textbook content. Please sign in to continue your learning journey.
          </p>
          <button
            onClick={() => history.push('/login')}
            style={{
              padding: '14px 32px',
              fontSize: '16px',
              fontWeight: 600,
              color: 'white',
              background: 'linear-gradient(135deg, #4299E1 0%, #667EEA 100%)',
              border: 'none',
              borderRadius: '12px',
              cursor: 'pointer',
              boxShadow: '0 4px 20px rgba(66, 153, 225, 0.4)',
              transition: 'all 0.3s ease',
            }}
            onMouseOver={(e) => {
              e.target.style.transform = 'translateY(-2px)';
              e.target.style.boxShadow = '0 6px 24px rgba(66, 153, 225, 0.6)';
            }}
            onMouseOut={(e) => {
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = '0 4px 20px rgba(66, 153, 225, 0.4)';
            }}
          >
            Sign In to Continue
          </button>
        </div>
      </div>
    );
  }

  return <DocPage {...props} />;
}

export default DocPageWithAuth;
