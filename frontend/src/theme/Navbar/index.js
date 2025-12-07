import React from 'react';
import Navbar from '@theme-original/Navbar';
import NavbarItems from '@site/src/components/NavbarItems';

export default function NavbarWrapper(props) {
    return (
        <>
            <Navbar {...props} />
            <style>{`
        .navbar__items--right {
          display: flex;
          align-items: center;
          gap: 1rem;
        }
      `}</style>
        </>
    );
}
