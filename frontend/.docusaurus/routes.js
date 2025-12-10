import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/login',
    component: ComponentCreator('/login', 'a8c'),
    exact: true
  },
  {
    path: '/signup',
    component: ComponentCreator('/signup', 'e02'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'a64'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', 'c20'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', 'ee7'),
            routes: [
              {
                path: '/docs/advanced/contributing',
                component: ComponentCreator('/docs/advanced/contributing', '69e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/advanced/faq',
                component: ComponentCreator('/docs/advanced/faq', '2e4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/advanced/humanoid-control',
                component: ComponentCreator('/docs/advanced/humanoid-control', '5bd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/advanced/manipulation',
                component: ComponentCreator('/docs/advanced/manipulation', 'ab4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/advanced/multi-robot',
                component: ComponentCreator('/docs/advanced/multi-robot', 'f93'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/advanced/navigation',
                component: ComponentCreator('/docs/advanced/navigation', '46f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ai-brain/intro',
                component: ComponentCreator('/docs/ai-brain/intro', '6f7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ai-brain/perception',
                component: ComponentCreator('/docs/ai-brain/perception', '63d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ai-brain/planning',
                component: ComponentCreator('/docs/ai-brain/planning', '9ca'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/digital-twin/intro',
                component: ComponentCreator('/docs/digital-twin/intro', '354'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/digital-twin/modeling',
                component: ComponentCreator('/docs/digital-twin/modeling', '9d0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/digital-twin/simulation-platforms',
                component: ComponentCreator('/docs/digital-twin/simulation-platforms', '385'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '61d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros2/architecture',
                component: ComponentCreator('/docs/ros2/architecture', '593'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros2/intro',
                component: ComponentCreator('/docs/ros2/intro', '864'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros2/nodes',
                component: ComponentCreator('/docs/ros2/nodes', '181'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros2/services',
                component: ComponentCreator('/docs/ros2/services', 'd53'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros2/topics',
                component: ComponentCreator('/docs/ros2/topics', '368'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/vla/architectures',
                component: ComponentCreator('/docs/vla/architectures', '617'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/vla/implementation',
                component: ComponentCreator('/docs/vla/implementation', 'd0f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/vla/intro',
                component: ComponentCreator('/docs/vla/intro', '4a8'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
