import React from 'react';
import { useAuth } from '../../context/AuthContext';

export default function NavbarItems() {
    const { user, logout, isAuthenticated } = useAuth();

    if (isAuthenticated && user) {
        return (
            <>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '1rem',
                    padding: '0.5rem 1rem',
                    backgroundColor: '#f0f9ff',
                    borderRadius: '6px',
                    border: '1px solid #bfdbfe'
                }}>
                    <span style={{
                        fontSize: '0.875rem',
                        color: '#1e40af',
                        fontWeight: '500'
                    }}>
                        {user.email}
                    </span>
                    <button
                        onClick={logout}
                        style={{
                            padding: '0.375rem 0.75rem',
                            backgroundColor: '#dc2626',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            fontSize: '0.875rem',
                            cursor: 'pointer',
                            fontWeight: '500',
                            transition: 'background-color 0.2s'
                        }}
                        onMouseOver={(e) => e.target.style.backgroundColor = '#b91c1c'}
                        onMouseOut={(e) => e.target.style.backgroundColor = '#dc2626'}
                    >
                        Logout
                    </button>
                </div>
            </>
        );
    }

    return null;
}
