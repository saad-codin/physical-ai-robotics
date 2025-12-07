---
sidebar_position: 2
---

# ROS 2 Nodes

A node is a single executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of ROS 2 applications, each performing a specific task or set of tasks.

## What is a Node?

In ROS 2, a node is a process that performs computation. Nodes are the primary building blocks of a ROS 2 program. Each node can perform a specific task and can communicate with other nodes to perform more complex operations.

### Key Characteristics

- **Encapsulation**: Each node encapsulates specific functionality
- **Communication**: Nodes communicate with each other through topics, services, and actions
- **Modularity**: Nodes can be developed and tested independently
- **Distributed**: Nodes can run on different machines in a network

## Creating Nodes

Nodes can be created in multiple programming languages including C++, Python, and others. Each node typically contains:

1. **Initialization**: Setting up the ROS 2 context
2. **Node Creation**: Creating the node instance
3. **Publishers/Subscribers**: Setting up communication interfaces
4. **Spin Loop**: Processing callbacks and events

### Python Example

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

### C++ Example

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0)
  {
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto message = std_msgs::msg::String();
    message.data = "Hello, world! " + std::to_string(count_++);
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;
};
```

## Node Lifecycle

ROS 2 nodes can be configured with a lifecycle to manage their state transitions:

- **Unconfigured**: Initial state after creation
- **Inactive**: Configured but not active
- **Active**: Fully operational
- **Finalized**: Node is being destroyed

This allows for more robust system management and coordination between nodes.

## Best Practices

- Keep nodes focused on a single responsibility
- Use appropriate quality of service (QoS) settings
- Implement proper error handling and logging
- Follow naming conventions for consistency
- Use parameters for configuration