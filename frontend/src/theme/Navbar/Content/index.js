import React, { useEffect } from 'react';
import NavbarContent from '@theme-original/Navbar/Content';
import { useAuth } from '@site/src/context/AuthContext';
import Link from '@docusaurus/Link';

export default function NavbarContentWrapper(props) {
    const { user, logout, isAuthenticated, loading } = useAuth();

    // Debug logging
    useEffect(() => {
        console.log('Navbar Auth State:', { isAuthenticated, user, loading });
    }, [isAuthenticated, user, loading]);

    return (
        <div style={{ display: 'flex', width: '100%', alignItems: 'center' }}>
            <NavbarContent {...props} />
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                {loading ? (
                    <div style={{ padding: '0.5rem' }}>
                        <span style={{ fontSize: '0.875rem', color: 'var(--ifm-color-primary)' }}>...</span>
                    </div>
                ) : isAuthenticated && user ? (
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.75rem',
                        padding: '0.4rem 1rem',
                        backgroundColor: 'var(--ifm-color-primary-lightest)',
                        borderRadius: '6px',
                        border: '1px solid var(--ifm-color-primary-lighter)'
                    }}>
                        <span style={{
                            fontSize: '0.875rem',
                            color: 'var(--ifm-color-primary-darkest)',
                            fontWeight: '500'
                        }}>
                            {user.email}
                        </span>
                        <button
                            onClick={logout}
                            style={{
                                padding: '0.25rem 0.75rem',
                                backgroundColor: '#dc2626',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                fontSize: '0.875rem',
                                cursor: 'pointer',
                                fontWeight: '500',
                                transition: 'background-color 0.2s'
                            }}
                            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#b91c1c'}
                            onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#dc2626'}
                        >
                            Logout
                        </button>
                    </div>
                ) : (
                    <>
                        <Link
                            to="/login"
                            style={{
                                padding: '0.375rem 0.75rem',
                                color: 'var(--ifm-color-primary)',
                                textDecoration: 'none',
                                fontSize: '0.875rem',
                                fontWeight: '500',
                                transition: 'color 0.2s'
                            }}
                        >
                            Login
                        </Link>
                        <Link
                            to="/signup"
                            style={{
                                padding: '0.375rem 0.75rem',
                                backgroundColor: 'var(--ifm-color-primary)',
                                color: 'white',
                                borderRadius: '4px',
                                textDecoration: 'none',
                                fontSize: '0.875rem',
                                fontWeight: '500',
                                transition: 'background-color 0.2s'
                            }}
                        >
                            Sign Up
                        </Link>
                    </>
                )}
            </div>
        </div>
    );
}
