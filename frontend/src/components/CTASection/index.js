import React from 'react';
import Link from '@docusaurus/Link';

export default function CTASection() {
  return (
    <section className="cta-section">
      <div className="cta-content">
        <h2 className="cta-title">Ready to Master Humanoid Robotics?</h2>
        <p className="cta-description">
          Join thousands of learners building the future of physical AI and robotics.
          Start your journey today with our comprehensive, AI-powered curriculum.
        </p>
        <div className="cta-buttons">
          <Link className="cta-button" to="/signup">
            Get Started Now
          </Link>
        </div>
      </div>
    </section>
  );
}
