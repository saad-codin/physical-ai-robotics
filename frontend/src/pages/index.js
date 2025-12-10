import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import StatsSection from '@site/src/components/StatsSection';
import CTASection from '@site/src/components/CTASection';
import Robot3D from '@site/src/components/Robot3D';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('robotics-header', 'futuristic-hero')}>
      <div className="hero-background">
        <div className="grid-lines"></div>
        <div className="floating-particles"></div>
      </div>
      <div className="container hero-container">
        <div className="hero-content">
          <h1 className="hero__title futuristic-title">
            <span className="title-glow">{siteConfig.title}</span>
          </h1>
          <p className="hero__subtitle futuristic-subtitle">{siteConfig.tagline}</p>
          <div className="hero-buttons">
            <Link
              className="button button--primary hero-button futuristic-button"
              to="/docs/intro">
              <span className="button-text">Start Learning</span>
              <span className="button-glow"></span>
            </Link>
          </div>
        </div>
        <div className="hero-robot">
          <Robot3D />
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="AI-Native Learning Platform for Physical AI & Humanoid Robotics">
      <HomepageHeader />
      <main>
        <StatsSection />
        <HomepageFeatures />
        <CTASection />
      </main>
    </Layout>
  );
}