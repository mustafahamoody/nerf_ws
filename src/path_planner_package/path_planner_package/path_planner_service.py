import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt

from nav_msgs.msg import OccupancyGrid # For Subscription
from nav_msgs.srv import GetPlan # For Service and Path Publishing
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# Path Planner Options: A* (A Star), RRT* (Rapidly-exploring Random Tree Star)
from nav.path_planners.path_planner_Astar import a_star
from nav.path_planners.path_planner_RRTstar import rrt_star

path_planner = rrt_star # Choose Path Planner to use: a_star (A*) or rrt_star (RRT*)


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_service')
        
        # Subscriber for occupancy grid
        # self.occupancy_grid_subscriber = self.create_subscription(OccupancyGrid, 'occupancy_grid_topic', self.occupancy_grid_callback, 10)
        
        # Service Server to publish path to controller (with start, goal, and headding parameters)
        self.serv = self.create_service(GetPlan, 'get_path', self.get_path_callback)

    # Function to visualise occ. grid and path with matplotlib
    def visualize_path(self, grid, path, start, goal):
        plt.imshow(grid, cmap='gray', origin='lower')
        plt.plot(start[1], start[0], 'ro', markersize=10)
        plt.plot(goal[1], goal[0], 'go', markersize=10)

        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2)
        
        plt.grid(True)
        plt.show()
   

    def get_path_callback(self, request, response):
        # Get grid from subscription
        # grid = np.array(msg.data).reshape(msg.info.height, msg.info.width) # Convert occupancy grid to numpy array
        
        # Empty grid
        # grid = np.zeros((10, 10))
        
        # # "Grid with box in middle"
        grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ])
        
        # Get start and goal from service client
        start = (float(request.start.pose.position.x), float(request.start.pose.position.y))
        goal = (float(request.goal.pose.position.x), float(request.goal.pose.position.y))

        self.get_logger().info(f'-------------Recived Path request from current position: {start} to goal: {goal}-------------')
        
        # Get path: Run path planner on grid
        path = path_planner(grid, start, goal) # Other Params (set to default): max_iter, step_size, goal_sample_rate, radius)

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

        # visualize the path
        self.visualize_path(grid, path, start, goal)
            
        return response

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlannerNode()
    rclpy.spin(planner)
    path_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()