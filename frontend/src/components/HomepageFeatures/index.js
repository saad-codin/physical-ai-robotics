import React from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Four-Module Curriculum',
    Svg: null, // Will be replaced with inline SVG
    description: (
      <>
        Comprehensive learning path covering ROS 2, Digital Twins, AI-Robot Brains,
        and Vision Language Action systems for humanoid robotics.
      </>
    ),
  },
  {
    title: 'AI-Powered Learning',
    Svg: null, // Will be replaced with inline SVG
    description: (
      <>
        Integrated RAG chatbot provides instant answers to questions about textbook content
        using semantic search and AI reasoning.
      </>
    ),
  },
  {
    title: 'Personalized Experience',
    Svg: null, // Will be replaced with inline SVG
    description: (
      <>
        Content adapts to your experience level, specialization, and learning preferences
        for optimal knowledge retention.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  // Simple inline SVG icons as fallback
  const getInlineSvg = (featureIndex) => {
    const svgProps = {
      width: 100,
      height: 100,
      viewBox: "0 0 24 24",
      fill: "#4299e1"
    };

    const icons = [
      // Mountain icon for Curriculum
      <svg key="mountain" {...svgProps}>
        <path d="M1 21h22L12 2 1 21zm2-2l9-16 9 16H3z" />
      </svg>,

      // Tree icon for AI Learning
      <svg key="tree" {...svgProps}>
        <path d="M16 10l4-6m-4 6l-4-6m4 6v6m4-3a4 4 0 11-8 0" />
        <path d="M3 21h18" stroke="#4299e1" strokeWidth="2" />
      </svg>,

      // React-like icon for Personalized Experience
      <svg key="react" {...svgProps}>
        <circle cx="12" cy="12" r="2" />
        <path d="M22 12c0 5.523-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2s10 4.477 10 10z" />
      </svg>
    ];

    return icons[featureIndex] || icons[0];
  };

  // Find the index of this feature in the list
  const featureIndex = FeatureList.findIndex(feature => feature.title === title);

  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        {Svg ? <Svg className={styles.featureSvg} role="img" /> : getInlineSvg(featureIndex)}
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}