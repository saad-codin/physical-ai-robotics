---
sidebar_position: 4
---

# ROS 2 Services

Services provide a request-response communication pattern in ROS 2. Unlike topics which are asynchronous, services provide synchronous communication where a client sends a request and waits for a response from a server.

## Service Communication

Service communication involves two parties:

- **Service Server**: Provides the service and processes requests
- **Service Client**: Makes requests to the service and receives responses

This pattern is useful for operations that require immediate responses or when you need to ensure a specific action is completed.

## Service Definition

Services are defined using `.srv` files that specify the request and response message types:

```
# Request part (before the --- separator)
string name
int32 age
---
# Response part (after the --- separator)
bool success
string message
```

## Creating Services

### Python Service Server Example

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main():
    rclpy.init()
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()
```

### Python Service Client Example

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    rclpy.init()
    minimal_client = MinimalClient()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(
        'Result of add_two_ints: %d' % response.sum)
    minimal_client.destroy_node()
    rclpy.shutdown()
```

## Service Commands

Common ROS 2 command-line tools for working with services:

```bash
# List all services
ros2 service list

# Show service information
ros2 service info /add_two_ints

# Call a service from command line
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"
```

## Advanced Service Concepts

### Service Types
ROS 2 provides various built-in service types in the `example_interfaces` package:

- `AddTwoInts`: Simple addition service
- `SetBool`: Boolean setting service
- `Trigger`: Simple trigger service
- `Empty`: Service with no parameters

### Custom Services
You can create custom service definitions by creating `.srv` files in your package's `srv` directory:

```
# In your_package/srv/CustomService.srv
string input_data
int32 config_value
---
bool success
string result_message
```

## When to Use Services

Services are appropriate for:

- Operations that require immediate results
- Configuration changes that need confirmation
- Actions that should complete before proceeding
- Simple request-response interactions
- Synchronous operations where order matters

## When Not to Use Services

Avoid services for:

- High-frequency data streaming (use topics instead)
- Fire-and-forget operations (use topics instead)
- Operations that might take a long time (use actions instead)
- Continuous monitoring tasks

## Best Practices

- Use services for operations that need guaranteed completion
- Implement proper error handling in service callbacks
- Consider timeouts when calling services
- Use appropriate service names that describe the action
- Document expected behavior and error conditions
- Consider using actions for long-running operations