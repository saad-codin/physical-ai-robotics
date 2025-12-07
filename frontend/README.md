# Physical AI & Humanoid Robotics Textbook - Frontend

This repository contains the frontend implementation of the AI-Native Physical AI & Humanoid Robotics Textbook, built with Docusaurus.

## Overview

The Physical AI & Humanoid Robotics Textbook is a comprehensive educational resource that integrates AI from the ground up, providing interactive learning experiences through our AI-powered chatbot and personalized content recommendations.

## Features

- **Four-Module Curriculum**: Complete learning path covering ROS 2, Digital Twins, AI-Robot Brains, and Vision Language Action systems
- **Interactive Components**: AI chatbot, code blocks with copy functionality, and simulator integration
- **AI-Powered Assistance**: RAG-based chatbot for instant Q&A and concept explanations
- **Personalized Learning**: Content adapts to experience level and learning preferences
- **Internationalization**: Support for multiple languages (English, Chinese, Spanish, French, German, Japanese, Korean)
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Project Structure

```
frontend/
├── blog/                    # Blog posts and updates
├── docs/                    # Main textbook documentation
│   ├── ros2/               # ROS 2 fundamentals module
│   ├── digital-twin/       # Digital twin & simulation module
│   ├── ai-brain/           # AI-robot brain module
│   ├── vla/                # Vision language action module
│   └── advanced/           # Advanced topics
├── src/                    # Custom React components
│   ├── components/         # Reusable components
│   │   ├── ChatKit/        # OpenAI ChatKit component
│   │   └── RoboticsCodeBlock/ # Interactive code blocks
│   ├── css/               # Custom styles
│   └── pages/             # Custom pages
├── static/                # Static assets
│   └── img/               # Images and icons
├── docusaurus.config.js   # Docusaurus configuration
├── sidebars.js            # Navigation sidebar configuration
└── package.json           # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The development server will start at `http://localhost:3000`.

## Development

### Adding New Content

To add new documentation pages:

1. Create a new `.md` file in the appropriate module directory
2. Add the page to the sidebar configuration in `sidebars.js`
3. Use the frontmatter to set the sidebar position:
```markdown
---
sidebar_position: 2
---

# Page Title
```

### Creating Components

Custom React components should be placed in `src/components/`:

```javascript
import React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

function MyComponent() {
  return (
    <BrowserOnly>
      {() => (
        // Component implementation
      )}
    </BrowserOnly>
  );
}

export default MyComponent;
```

### Styling

Custom styles can be added to `src/css/custom.css`. The project uses the default Docusaurus styling with additional robotics-specific customizations.

## Components

### ChatKit Integration

The ChatKit component provides interactive learning assistance with RAG capabilities:

```jsx
import ChatKitComponent from '@site/src/components/ChatKit';

function MyPage() {
  return (
    <div>
      <ChatKitComponent />
    </div>
  );
}
```

### Robotics Code Blocks

Interactive code blocks with copy functionality and simulator integration:

```jsx
import RoboticsCodeBlock from '@site/src/components/RoboticsCodeBlock';

function MyPage() {
  return (
    <RoboticsCodeBlock title="ROS 2 Publisher Example" language="python">
{`import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0`}
    </RoboticsCodeBlock>
  );
}
```

## Internationalization

The textbook supports multiple languages. To add content in a new language:

1. Create a new directory in `i18n/[locale-code]/docusaurus-plugin-content-docs/`
2. Copy the existing documentation structure
3. Translate the content
4. Add the locale to `docusaurus.config.js`

## Building for Production

To build the static site for production:

```bash
npm run build
```

The built site will be in the `build/` directory.

To serve the built site locally for testing:

```bash
npm run serve
```

## Deployment

The site can be deployed to various platforms:

- **GitHub Pages**: Use the `gh-pages` branch
- **Netlify**: Deploy the `build/` directory
- **Vercel**: Deploy the project root
- **Any static hosting**: Serve the `build/` directory

## API Integration

The frontend is designed to integrate with the backend API for features like:

- AI chatbot responses
- Personalized content recommendations
- User progress tracking
- Exercise submissions

API endpoints are configured in the AI chatbot component and can be updated as needed.

## Contributing

We welcome contributions to the textbook! Please see our [Contributing Guide](./docs/advanced/contributing.md) for details on how to contribute content, examples, and improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the repository or contact the maintainers.