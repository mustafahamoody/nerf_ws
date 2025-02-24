import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt

from nav_msgs.msg import OccupancyGrid # For Subscription
from nav_msgs.srv import GetPlan # For Service and Path Publishing
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# Path Planner Options: A* (A Star), RRT* (Rapidly-exploring Random Tree Star)
from path_planner_package.path_planners.path_planner_Astar import a_star
from path_planner_package.path_planners.path_planner_RRTstar import rrt_star

path_planner = rrt_star # Choose Path Planner to use: a_star (A*) or rrt_star (RRT*)


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_service')
        
        # Subscriber for occupancy grid
        self.occupancy_grid_subscriber = self.create_subscription(OccupancyGrid, 'occupancy_grid_topic', self.get_occupancy_grid, 10)
        
        # Service Server to publish path to controller (with start, goal, and headding parameters)
        self.service = self.create_service(GetPlan, 'get_path', self.get_path_callback)

        # Object to store occupancy grid
        self.grid = None

    def get_occupancy_grid(self, msg):
        # Get grid from subscription
        width, height = msg.info.width, msg.info.height
        data = np.array(msg.data).reshaper((height, width))
        self.grid = np.where(data == -1, 100, data)  # Replace unknown (-1) with high cost (100)
        self.get_logger().info('Occupancy grid received')

    def get_path_callback(self, request, response):
        
        # Empty grid
        # grid = np.zeros((10, 10))
        
        # # "Grid with box in middle"
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
        
        # Get start and goal from service client
        start = (float(request.start.pose.position.x), float(request.start.pose.position.y))
        goal = (float(request.goal.pose.position.x), float(request.goal.pose.position.y))

        self.get_logger().info(f'-------------Recived Path request from current position: {start} to goal: {goal}-------------')
        
        # Get path: Run path planner on grid
        path = path_planner(self.grid, start, goal) # Other Params (set to default): max_iter, step_size, goal_sample_rate, radius)

        # Create Response and Response plan (path) objects
        response = GetPlan.Response()
        response.plan = Path()

        # Response Header
        response.plan.header.stamp = self.get_clock().now().to_msg()
        response.plan.header.frame_id = 'map'

        if path:
            for x, y in path:
                pose = PoseStamped()
                pose.header = response.plan.header

                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.position.z = 0.0

                response.plan.poses.append(pose)
         
        return response

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlannerNode()
    rclpy.spin(planner)
    path_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()