---
sidebar_position: 2
---

# ROS 2 Architecture

Understanding the architecture of ROS 2 is crucial for developing effective robotic applications. The architecture is built around several core concepts and technologies.

## DDS (Data Distribution Service)

ROS 2 uses DDS as its underlying communication middleware. DDS provides:

- **Decentralized Architecture**: No single point of failure
- **Quality of Service (QoS)**: Configurable reliability and performance settings
- **Language Independence**: Support for multiple programming languages
- **Platform Independence**: Works across different operating systems

## Nodes and Processes

In ROS 2, nodes are individual processes that communicate with each other:

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World'
        self.publisher.publish(msg)
```

## Communication Patterns

### Topics (Publish/Subscribe)
- Asynchronous communication
- Multiple publishers and subscribers possible
- Data is broadcast to all subscribers

### Services (Request/Response)
- Synchronous communication
- One-to-one communication
- Request/response pattern

### Actions (Goal/Feedback/Result)
- For long-running tasks
- Support for cancellation and feedback
- Goal-Feedback-Result pattern

## Package Structure

ROS 2 packages typically contain:

- `package.xml`: Package manifest
- `CMakeLists.txt`: Build configuration for C++
- `setup.py`: Build configuration for Python
- `src/`: Source code
- `include/`: Header files
- `launch/`: Launch files
- `config/`: Configuration files

## Communication Middleware Implementation (CMI)

ROS 2 allows different DDS implementations to be used:

- **Fast DDS**: Default implementation from eProsima
- **Cyclone DDS**: From Eclipse
- **RTI Connext DDS**: Commercial implementation

This flexibility allows ROS 2 to be used in various environments with different requirements.