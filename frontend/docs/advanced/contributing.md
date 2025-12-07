---
sidebar_position: 1
---

# Contributing to the Textbook

We welcome contributions to the Physical AI & Humanoid Robotics Textbook! This guide will help you get started with contributing content, examples, and improvements to the textbook.

## Getting Started

### Prerequisites

Before contributing, make sure you have:

- Basic understanding of robotics concepts
- Experience with ROS 2, simulation platforms, or AI systems
- Familiarity with Git and GitHub workflows
- Docusaurus knowledge (optional but helpful)

### Setting Up Your Environment

1. Fork the repository
2. Clone your fork locally
3. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

4. Start the development server:
   ```bash
   npm start
   ```

## Types of Contributions

### Content Contributions

We welcome contributions in the form of:

- **New documentation pages**: Adding new topics or expanding existing ones
- **Code examples**: Practical implementations and demonstrations
- **Exercises**: Hands-on activities for learners
- **Case studies**: Real-world applications and examples
- **Tutorials**: Step-by-step guides for complex topics

### Technical Contributions

- **Bug fixes**: Correcting errors in documentation or code
- **Feature enhancements**: Improving existing functionality
- **Performance improvements**: Optimizing code or content
- **Accessibility improvements**: Making content more accessible

### Community Contributions

- **Translations**: Translating content into other languages
- **Reviews**: Reviewing and improving existing content
- **Answers**: Helping other learners in the community

## Content Guidelines

### Writing Style

- Use clear, concise language
- Explain concepts with practical examples
- Include relevant code snippets and diagrams
- Follow the existing structure and formatting

### Technical Accuracy

- Verify all code examples work as described
- Include appropriate citations and references
- Test all instructions and procedures
- Ensure examples are up-to-date with current technologies

### Code Examples

When contributing code examples:

```python
# Use clear variable names
robot_speed = 0.5  # meters per second

# Include comments explaining complex logic
if sensor_data.distance < threshold:
    # Stop the robot to avoid collision
    robot.stop()
```

## Development Workflow

### Creating a New Branch

```bash
git checkout -b feature/descriptive-branch-name
```

### Making Changes

1. Make your changes following the guidelines
2. Test your changes locally
3. Add or update documentation as needed

### Committing Changes

```bash
git add .
git commit -m "Add: detailed explanation of ROS 2 lifecycle nodes"
```

Use descriptive commit messages that explain what was changed and why.

### Submitting a Pull Request

1. Push your changes to your fork
2. Create a pull request to the main repository
3. Fill out the pull request template
4. Link any relevant issues

## Code of Conduct

Please follow our Code of Conduct in all interactions:

- Be respectful and inclusive
- Provide constructive feedback
- Focus on technical content
- Help others learn and grow

## Review Process

All contributions go through a review process:

1. **Technical Review**: Checking for accuracy and completeness
2. **Style Review**: Ensuring consistency with textbook style
3. **Testing**: Verifying code examples and instructions
4. **Integration**: Merging approved changes

## Getting Help

### Questions

If you have questions about contributing:

- Check the existing documentation
- Open an issue for clarification
- Join our community discussions

### Feedback

We value feedback on all contributions. Be prepared to:

- Address review comments
- Iterate on your contributions
- Collaborate with other contributors

## Recognition

Contributors are recognized in:

- The contributors section of the textbook
- Release notes for significant contributions
- Acknowledgments in related materials

## Advanced Contribution Areas

### Research Integration

We encourage contributions that integrate cutting-edge research:

- Recent papers and findings
- Experimental results
- Research methodologies

### Industry Applications

Real-world applications and case studies:

- Industrial robotics implementations
- Commercial robotics solutions
- Field deployment experiences

### Educational Enhancements

Improvements to the learning experience:

- Interactive elements
- Assessment tools
- Learning analytics

Thank you for contributing to the Physical AI & Humanoid Robotics Textbook! Your contributions help make robotics education more accessible and effective for learners worldwide.