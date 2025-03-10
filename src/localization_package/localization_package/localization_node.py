import rclpy
from rclpy import Node
from nav_msgs.msg import Pose

from localization_package.localization_package.localize import PositionEstimator

class LocalizationNode(Node):
    def __init__(self):
        super().__init__('localization_node')

        # Service Server to get send pose (x, y, z) and theta, determined by localization algorithm
        self.service = self.create_service(Pose, 'pose', self.localize_callback)

    def localize_callback(self, response):
        # Get pose from localization algorithm
        self.get_logger().info('-------------Recived Position Query-------------')

        position = ...
    
        


        return response
        

def main(args=None):
    rclpy.init(args=args)
    localizer = LocalizationNode()
    rclpy.spin(localizer)
    localizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()