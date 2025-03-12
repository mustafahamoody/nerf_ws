import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import yaml

# NeRF Functions
from nerf_config.libs.nerf.utils import *
from nerf_config.libs.nerf.provider import NeRFDataset
from nerf_config.libs.nerf.network import NeRFNetwork
from nerf_config.config.model_options import ModelOptions

# Helper: convert yaw (rotation about z) into a 3x3 rotation matrix.
def yaw_to_matrix(yaw):
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    R = torch.tensor([[cos_y, -sin_y, 0],
                      [sin_y, cos_y, 0],
                      [0,     0,     1]], dtype=yaw.dtype, device=yaw.device)
    return R

# Simple keypopint detector using SIFT
def find_keypoints(camera_image, render=False):
    img = np.copy(camera_image)

    # Use Scale-Invariant Feature Transform (SIFT) to detect keypoints
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)

    if render:
        # Draw keypoints on image
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feature_img = cv2.drawKeypoints(img_gray, keypoints, img)
    else:
        feature_img = None

    # Extract (x,y) coords. of keypoints on the image (as a numpy int array)
    keypoint_coords = np.array([keypoint.pt for keypoint in keypoints]).astype(int)
    
    # Remove duplicate keypoints
    keypoint_coords = np.unique(keypoint_coords, axis=0)

    extras = {'features': feature_img}

    return keypoint_coords, extras


# Pose optimizer for a ground rover: only x, y and yaw are optimized.
class PoseOptimizer():
    def __init__(self, render_fn, get_rays_fn, learning_rate=0.01, n_iters=500, batch_size=1024, 
                 kernel_size=3, dilate_iter=2, render_viz=False, fixed_z=0.0):
        """
        render_fn: function taking rays (origins, directions) and returning a dict with key 'image'
        get_rays_fn: function taking a 4x4 camera pose matrix and returning dict with 'rays_o' and 'rays_d'
        lrate: learning rate for optimization
        num_iters: number of gradient descent iterations
        batch_size: number of pixels (from feature regions) used per iteration
        fixed_z: fixed z coordinate (camera height)
        render_viz: if True, display intermediate renders
        """

        self.render_fn = render_fn
        self.get_rays = get_rays_fn

        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.dilate_iter = dilate_iter
        self.render_viz = render_viz


    def estimate_pose(self, camera_image):
        """
        sensor_image: RGB image as a numpy array (range 0...255)
        Returns: (x, y, z) translation tuple, yaw (radians), and the loss history.
        """

        H, W, _ = camera_image.shape
        camera_image = (camera_image / 255).astype(np.float32)
        camera_image_t = torch.tensor(camera_image).permute(2, 0, 1).unsqueeze(0).cuda()


        # Detect Keypoints
        keypoints, extras = find_keypoints(camera_image, render=self.render_viz)
        if keypoints.shape[0] == 0:
            print("No Features Found in Image")
            return None
    

        # Create mask from keypoints and dilate it
        intrest_mask = np.zeros((H, W), dtype=np.uint8)
        # Keypoints are (x,y); use y for row and x for column.
        intrest_mask[keypoints[:, 1], keypoints[:, 0]] = 1 
        
        intrest_mask = cv2.dilate(intrest_mask, np.ones((self.kernel_size, self.kernel_size), np.uint8),
                                    iterations=self.dilate_iter)
        intrest_idxs = np.argwhere(intrest_mask > 0)


        # Initlaize x, y and yaw params -- As torch tenors with Grad. Used for pose optimization
        pose_params = torch.zeros(4, dtype=torch.float32, device='cuda', requires_grad=True)
        optimizer = torch.optim.Adam([pose_params], lr=self.learning_rate)
        losses = []


        # Optimization Loop
        for itter in range(1, self.n_iters + 1):

            # Unpack parameters
            x, y, z, yaw = pose_params[0], pose_params[1], pose_params[2], pose_params[3]
            R = yaw_to_matrix(yaw)

            # Build the 4x4 pose matrix
            pose_matrix = torch.eye(4, dtype=torch.float32, device='cuda', requires_grad=True) # Create Identity Matrix
            pose_matrix[:3, :3] = R
            pose_matrix[:3, 3] = torch.tensor([x, y, z], device='cuda')

            # Generate rays for current pose
            rays = self.get_rays(pose_matrix.unsqueeze(0))
            rays_o = rays["rays_o"].reshape(H, W, -1)
            rays_d = rays["rays_d"].reshape(H, W, -1)

            # Sample batch of pixels from the intrest reigon
            if intrest_idxs.shape[0] < self.batch_size:
                batch_idxs = intrest_idxs
            else:
                idxs = np.random.choice(intrest_idxs.shape[0], self.batch_size, replace=False)
                batch_idxs = intrest_idxs[idxs]
            
            batch_idxs = torch.tensor(batch_idxs, dtype=torch.long, device='cuda')
            batch_y = batch_idxs[:, 0]
            batch_x = batch_idxs[:, 1]

            rays_o_batch = rays_o[batch_y, batch_x].unsqueeze(0)   # Shape: [1, N, 3]
            rays_d_batch = rays_d[batch_y, batch_x].unsqueeze(0)   # Shape: [1, N, 3]

            # Render batch from the current pose
            output = self.render_fn(rays_o_batch, rays_d_batch)
            rendered_rgb = output["image"].reshape(-1, 3)
            camera_rgb = camera_image_t[0, :, batch_y, batch_x].permute(1, 0)

            # Compute Loss using MSE (mean-squared error) loss
            loss = torch.nn.functional.mse_loss(rendered_rgb, camera_rgb)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if itter <= 3 or itter % 50 == 0:
                print(f'Itteration {itter}, Loss: {loss.item()}')
                if self.render_viz:
                    full_rays = self.get_rays(pose_matrix.unsqueeze(0))
                    output_full = self.render_fn(full_rays["rays_o"], full_rays["rays_d"])
                    full_render = output_full["image"].reshape(H, W, 3).detach().cpu().numpy()
                    plt.figure(figsize=(8, 8))
                    plt.imshow(full_render)
                    plt.title(f'Itteration {itter}')
                    plt.show()
                    plt.close()

        final_pose = pose_params.detach().cpu().numpy()
        final_translation = (final_pose[0], final_pose[1], final_pose[2])
        final_yaw = final_pose[3]
        return final_translation, final_yaw, losses


########################### Load NERF Model Config. ###########################

def load_config(file_path):
    """Load YAML configuration file with environment variable expansion."""
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        def expand_vars(item):
            if isinstance(item, str):
                return os.path.expandvars(item)
            elif isinstance(item, dict):
                return {key: expand_vars(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [expand_vars(elem) for elem in item]
            else:
                return item

        config = expand_vars(config)
        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file: {e}")

########################### END Load NERF Model Config. ###########################

class localize():
    def __init__(self):
        # Get config paths from environment variables with error checking
        model_config_path = os.environ.get('MODEL_CONFIG_PATH')
        trainer_config_path = os.environ.get('TRAINER_CONFIG_PATH')
        
        if not model_config_path:
            raise EnvironmentError("MODEL_CONFIG_PATH environment variable must be set")
        if not trainer_config_path:
            raise EnvironmentError("TRAINER_CONFIG_PATH environment variable must be set")
            
        # Load configurations
        try:
            self.config_model = load_config(model_config_path)
            self.config_trainer = load_config(trainer_config_path)

        except Exception as e:
            self.get_logger().error(f"Failed to load configurations: {e}")
            raise

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Initialize the NeRFNetwork model
        self.model = NeRFNetwork(
            encoding=self.config_model['model']['encoding'],
            bound=self.config_model['model']['bound'],
            cuda_ray=self.config_model['model']['cuda_ray'],
            density_scale=self.config_model['model']['density_scale'],
            min_near=self.config_model['model']['min_near'],
            density_thresh=self.config_model['model']['density_thresh'],
            bg_radius=self.config_model['model']['bg_radius'],
        )
        self.model.eval()  # Set the model to evaluation mode

        self.metrics = [PSNRMeter(),]
        self.criterion = torch.nn.MSELoss(reduction='none')

        # Initialize the Trainer (this loads weights from a checkpoint)
        self.trainer = Trainer(
            'ngp',
            opt=ModelOptions.opt(),
            model=self.model,
            device=self.device,
            workspace=self.config_trainer['trainer']['workspace'],
            criterion=self.criterion,
            fp16=self.config_model['model']['fp16'],
            metrics=self.metrics,
            use_checkpoint=self.config_trainer['trainer']['use_checkpoint'],
        )

        self.dataset = NeRFDataset(ModelOptions.opt(), device=self.device, type='test')  # Importing dataset in order to get the same camera intrinsics as training    
        self.render_fn = lambda rays_o, rays_d: self.model.render(rays_o, rays_d, staged=True, bg_color=1., perturb=False, **vars(ModelOptions.opt()))  # Function to Render Image
        self.get_rays_fn = lambda pose: get_rays(pose, self.dataset.intrinsics, self.dataset.H, self.dataset.W)  # Function to Generate Render rays


    def run(self, camera_image):
        optimizer = PoseOptimizer(self.render_fn, self.get_rays_fn, learning_rate=0.005,
                                            n_iters=500, render_viz=True)
        
        result = optimizer.estimate_pose(camera_image) # Run Pose Optimizer on Image
        
        if result is not None:
            final_translation, final_yaw, losses = result
            # final_translation is a tuple (x, y, z)
            print("Final translation (x, y, z):", final_translation)
            print("Final yaw (radians):", final_yaw)





################## TEST ##################
# Load your sensor image (ensure it is in RGB).
camera_image = cv2.imread("1.png")
camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

localize().run(camera_image)