// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'ROS 2 Fundamentals',
      items: [
        'ros2/intro',
        'ros2/architecture',
        'ros2/nodes',
        'ros2/topics',
        'ros2/services',
      ],
    },
    {
      type: 'category',
      label: 'Digital Twin & Simulation',
      items: [
        'digital-twin/intro',
        'digital-twin/simulation-platforms',
        'digital-twin/modeling',
      ],
    },
    {
      type: 'category',
      label: 'AI-Robot Brain',
      items: [
        'ai-brain/intro',
        'ai-brain/perception',
        'ai-brain/planning',
      ],
    },
    {
      type: 'category',
      label: 'Vision Language Action (VLA)',
      items: [
        'vla/intro',
        'vla/architectures',
        'vla/implementation',
      ],
    },
    {
      type: 'category',
      label: 'Advanced Topics',
      items: [
        'advanced/contributing',
        'advanced/faq',
        'advanced/humanoid-control',
        'advanced/manipulation',
        'advanced/navigation',
        'advanced/multi-robot',
      ],
    },
  ],
};

module.exports = sidebars;