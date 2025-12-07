import React from 'react';
import NavbarItem from '@theme-original/NavbarItem';
import { useAuth } from '@site/src/context/AuthContext';

export default function NavbarItemWrapper(props) {
    const { isAuthenticated } = useAuth();

    // Hide Login and Sign Up links when user is authenticated
    if (isAuthenticated && (props.to === '/login' || props.to === '/signup')) {
        return null;
    }

    return <NavbarItem {...props} />;
}
