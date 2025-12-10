import React from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Four-Module Curriculum',
    icon: '📚',
    description: (
      <>
        Comprehensive learning path covering ROS 2, Digital Twins, AI-Robot Brains,
        and Vision Language Action systems for humanoid robotics.
      </>
    ),
  },
  {
    title: 'AI-Powered Learning',
    icon: '🤖',
    description: (
      <>
        Integrated RAG chatbot provides instant answers to questions about textbook content
        using semantic search and AI reasoning.
      </>
    ),
  },
  {
    title: 'Personalized Experience',
    icon: '✨',
    description: (
      <>
        Content adapts to your experience level, specialization, and learning preferences
        for optimal knowledge retention.
      </>
    ),
  },
];

function Feature({icon, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="feature-card">
        <div className="feature-icon text--center">
          <span style={{ fontSize: '3rem' }}>{icon}</span>
        </div>
        <div className="text--center">
          <Heading as="h3" className="feature-title">{title}</Heading>
          <p className="feature-description">{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className="features-section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">Why Choose Our Textbook?</h2>
          <p className="section-subtitle">
            Experience the next generation of robotics education with AI-enhanced learning
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}