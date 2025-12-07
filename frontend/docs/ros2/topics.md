---
sidebar_position: 3
---

# ROS 2 Topics

Topics are named buses over which nodes exchange messages. They provide a way for nodes to communicate with each other using a publish-subscribe pattern.

## Topic Communication

Topics use a publish-subscribe communication pattern where:

- **Publishers** send messages to a topic
- **Subscribers** receive messages from a topic
- Multiple publishers and subscribers can use the same topic
- Communication is asynchronous and decoupled

## Quality of Service (QoS)

ROS 2 provides Quality of Service settings to control how messages are delivered:

### Reliability Policy
- **Reliable**: All messages are delivered, possibly with retries
- **Best Effort**: Messages are delivered without guarantees

### Durability Policy
- **Transient Local**: Late-joining subscribers receive the last message
- **Volatile**: No messages are stored for late joiners

### History Policy
- **Keep Last**: Maintain a fixed number of messages
- **Keep All**: Maintain all messages (use with caution)

## Creating Publishers and Subscribers

### Python Publisher Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):

    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
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

### Python Subscriber Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):

    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
```

## Topic Commands

Common ROS 2 command-line tools for working with topics:

```bash
# List all topics
ros2 topic list

# Show topic information
ros2 topic info /chatter

# Echo messages from a topic
ros2 topic echo /chatter

# Publish a message to a topic
ros2 topic pub /chatter std_msgs/String "data: 'Hello'"
```

## Advanced Topic Concepts

### Topic Remapping
Topics can be remapped at runtime to change communication patterns:

```python
# In launch files or via command line
# Remap /original_topic to /new_topic
```

### Topic Namespaces
Use namespaces to organize topics in complex systems:

```
/robot1/sensors/lidar
/robot1/control/cmd_vel
/robot2/sensors/lidar
/robot2/control/cmd_vel
```

## Best Practices

- Use descriptive and consistent topic names
- Choose appropriate QoS settings based on application requirements
- Monitor topic rates to avoid overwhelming the system
- Use appropriate message types for the data being transmitted
- Consider message size and frequency for network performance