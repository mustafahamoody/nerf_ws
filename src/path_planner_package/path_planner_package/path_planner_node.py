import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt

from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# Path Planner Options: A* (A Star), RRT* (Rapidly-exploring Random Tree Star)
from path_planner_package.path_planners.path_planner_Astar import a_star
from path_planner_package.path_planners.path_planner_RRTstar import rrt_star

# # Get start and goal positions from user
# start = input("Enter start location x, y: ").strip() or "0, 0"
# start = tuple(map(int, start.split(',')))

# goal = input("Enter goal location x, y: ").strip() or "9, 9"
# goal = tuple(map(int, goal.split(',')))


# Choose Path Planner to use: a_star (A*) or rrt_star (RRT*)
path_planner = rrt_star

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        
        # Publisher for path
        self.path_publisher_ = self.create_publisher(Path, 'path', 10)
        self.path_timer = self.create_timer(1.0, self.path_publisher_callback)
        # Make sure path is published only once
        self.path_published = False

        # Subscriber for occupancy grid
        self.occupancy_grid_subscriber = self.create_subscription(OccupancyGrid, 'occupancy_grid_topic', self.get_occupancy_grid, 10)

        # Object to store occupancy grid
        self.grid = None

    def get_occupancy_grid(self, msg):
        # Get grid from subscription
        width, height = msg.info.width, msg.info.height
        data = np.array(msg.data).reshape((height, width))

        if not self.grid:
            self.get_logger().error('No Occupancy Grid Received')
            return
        
        self.grid = np.where(data == -1, 100, data)  # Replace unknown (-1) with high cost (100)
        self.get_logger().info('Occupancy grid received')


    def path_publisher_callback(self): #msg):
        
        if self.path_published == True:
            return
        
        # grid = np.array(msg.data).reshape(msg.info.height, msg.info.width) # Convert occupancy grid to numpy array
        # grid = np.zeros((10, 10)) # Example grid
        
        # "Env with Box in Middle"
        # grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        #                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        #                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        #                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                 ])

        # Set to disable user input
        start = (0, 0) 
        goal = (9, 9)

        # Call path planner to create path
        path = path_planner(self.grid, start, goal) # Other Params (set to default) max_iter=100000, step_size=1.0, goal_sample_rate=0.3, radius=2.0)

        if path:
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'map'

            for x, y in path:
                pose = PoseStamped()
                pose.header = path_msg.header

                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.position.z = 0.0

                path_msg.poses.append(pose)
            self.path_publisher_.publish(path_msg)

            # visualize the path
            self.path_published = True


def main(args=None):
    rclpy.init(args=args)
    path_planner_node = PathPlannerNode()
    rclpy.spin(path_planner_node)
    path_planner_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()