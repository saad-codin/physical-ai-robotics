import React from 'react';
import { useAuth } from '../../context/AuthContext';
import Link from '@docusaurus/Link';

const AuthNavbar = () => {
  const { user, logout, isAuthenticated } = useAuth();

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
      {isAuthenticated ? (
        <>
          <span style={{ color: '#4a5568' }}>
            Welcome, {user?.first_name || user?.email || 'User'}
          </span>
          <button
            onClick={logout}
            style={{
              padding: '0.25rem 0.75rem',
              backgroundColor: '#e53e3e',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.875rem',
            }}
          >
            Logout
          </button>
        </>
      ) : (
        <>
          <Link
            to="/login"
            style={{
              padding: '0.25rem 0.75rem',
              backgroundColor: '#4299e1',
              color: 'white',
              textDecoration: 'none',
              borderRadius: '4px',
              fontSize: '0.875rem',
            }}
          >
            Login
          </Link>
          <Link
            to="/signup"
            style={{
              padding: '0.25rem 0.75rem',
              backgroundColor: '#48bb78',
              color: 'white',
              textDecoration: 'none',
              borderRadius: '4px',
              fontSize: '0.875rem',
            }}
          >
            Sign Up
          </Link>
        </>
      )}
    </div>
  );
};

export default AuthNavbar;