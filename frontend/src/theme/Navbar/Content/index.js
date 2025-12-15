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
        <>
            <NavbarContent {...props} />
            <div className="custom-navbar-auth-items">
                {loading ? (
                    <div style={{ padding: '0.5rem' }}>
                        <span style={{ fontSize: '0.875rem', color: 'var(--ifm-color-primary)' }}>...</span>
                    </div>
                ) : isAuthenticated && user ? (
                    <div className="navbar-auth-user-info">
                        <span className="navbar-auth-email">
                            {user.email}
                        </span>
                        <button
                            onClick={logout}
                            className="navbar-auth-logout-btn"
                            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#b91c1c'}
                            onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#dc2626'}
                        >
                            Logout
                        </button>
                    </div>
                ) : (
                    <>
                        <Link to="/login" className="navbar-auth-link">
                            Login
                        </Link>
                        <Link to="/signup" className="navbar-auth-signup-btn">
                            Sign Up
                        </Link>
                    </>
                )}
            </div>
        </>
    );
}
