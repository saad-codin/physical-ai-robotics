import React from 'react';

const stats = [
  { number: '4', label: 'Core Modules' },
  { number: '100+', label: 'Topics Covered' },
  { number: '50+', label: 'Code Examples' },
  { number: '24/7', label: 'AI Assistant' },
];

export default function StatsSection() {
  return (
    <section className="stats-section">
      <div className="container">
        <div className="stats-grid">
          {stats.map((stat, idx) => (
            <div key={idx} className="stat-card">
              <div className="stat-number">{stat.number}</div>
              <div className="stat-label">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
