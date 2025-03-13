import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import yaml
import os

# NeRF Functions
from nerf_config.libs.nerf.utils import *
from nerf_config.libs.nerf.provider import NeRFDataset
from nerf_config.libs.nerf.network import NeRFNetwork
from nerf_config.config.model_options import ModelOptions


# Image Keypoints detector using SIFT
def find_keypoints(camera_image, render=False):
    image = np.copy(camera_image)

    if image is None:
        print("No Image Recieved")
    # else:
    #     print(f'-------------------------------{image.shape, image.dtype}-------------------------------')

    if image.dtype != 'uint8':
        # Normalize to [0, 255] and convert to uint8
        image = cv2.convertScaleAbs(image, alpha=(255.0 / np.max(image)))
    
    # Convert image to grayscale -- SIFT works best with grayscale images
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Use Scale-Invariant Feature Transform (SIFT) to detect keypoints
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)

    if render:
        # Draw keypoints on image
        feature_image = cv2.drawKeypoints(image_gray, keypoints, image)
    else:
        feature_image = None

    # Extract (x,y) coords. of keypoints on the image (as a numpy int array)
    keypoint_coords = np.array([keypoint.pt for keypoint in keypoints]).astype(int)
    
    # Remove duplicate keypoints
    keypoint_coords = np.unique(keypoint_coords, axis=0)

    extras = {'features': feature_image}

    return keypoint_coords, extras


# Pose Coordinate Optimizer using NeRF
class PoseOptimizer():
    def __init__(self, render_fn, get_rays_fn, learning_rate=0.01, n_iters=500, batch_size=2048, 
                 kernel_size=3, dilate_iter=2, render_viz=False, fixed_z=0.0):
        """
        render_fn: function taking rays (origins, directions) and returning a dict with key 'image'
        get_rays_fn: function taking a 4x4 camera pose matrix and returning dict with 'rays_o' and 'rays_d'
        learning_rate: learning rate for optimization
        n_iters: number of gradient descent iterations
        batch_size: number of pixels (from feature regions) used per iteration
        fixed_z: fixed z coordinate (camera height) -- For ground rovers
        save_render: if True, save intermediate renders to file
        """
        self.render_fn = render_fn
        self.get_rays = get_rays_fn
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.dilate_iter = dilate_iter
        self.render_viz = render_viz
        self.fixed_z = fixed_z


    def estimate_pose(self, camera_image):
        """
        camera_image: RGB image as a numpy array (range 0...255)
        Returns: (x, y, z) translation tuple, yaw (radians), and the loss history.
        """
        H, W, _ = camera_image.shape
        camera_image = (camera_image / 255).astype(np.float32)
        # Move tensor to CUDA and permute dimensions to match expected format
        camera_image_t = torch.tensor(camera_image, device='cuda').permute(2, 0, 1).unsqueeze(0)
        
        # Detect Keypoints
        keypoints, extras = find_keypoints(camera_image, render=self.render_viz)
        if keypoints.shape[0] == 0:
            print("No Features Found in Image")
            return None
        
        # Create mask from keypoints and dilate it
        interest_mask = np.zeros((H, W), dtype=np.uint8)
        interest_mask[keypoints[:, 1], keypoints[:, 0]] = 1 
        
        interest_mask = cv2.dilate(interest_mask, np.ones((self.kernel_size, self.kernel_size), np.uint8),
                                    iterations=self.dilate_iter)
        interest_idxs = np.argwhere(interest_mask > 0)

        # Create optimizable (trainable) pose parameters -- with small non-zero values to avoid local minima
        pose_params = torch.tensor([0.1, 0.1, self.fixed_z, 0.05], device='cuda', requires_grad=True)

        # Use Adam optimizer for poses
        optimizer = torch.optim.Adam([pose_params], lr=self.learning_rate)
        # Add learning rate schediler for better converence
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

        # Tracking loss history
        losses = []

        if self.render_viz:
            # Create render_viz directory to store NeRF Renders
            render_dir = os.path.join(os.getcwd(), "render_viz")
            os.makedirs(render_dir, exist_ok=True)

        
        # Pose Optimization Loop
        for iter in range(1, self.n_iters + 1):
            optimizer.zero_grad()

            # Extract Parameters
            x, y, z, yaw = pose_params[0], pose_params[1], pose_params[2], pose_params[3]

            # Create rotation matrix for yaw
            cos_y = torch.cos(yaw)
            sin_y = torch.sin(yaw)
            R = torch.zeros((3, 3), device='cuda')
            R[0, 0] = cos_y
            R[0, 1] = -sin_y
            R[1, 0] = sin_y
            R[1, 1] = cos_y
            R[2, 2] = 1.0

            # Create translation vector
            t = torch.zeros((3, 1), device='cuda')
            t[0, 0] = x
            t[1, 0] = y
            t[2, 0] = z

            # Build pose matrix (top 3x4 part)
            top_rows = torch.cat([R, t], dim=1)

            # Add homogeneous row
            bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device='cuda')
            pose_matrix = torch.cat([top_rows, bottom_row], dim=0).unsqueeze(0) # Add batch dim


            # Get rays for entire image
            rays = self.get_rays(pose_matrix)
            rays_o = rays["rays_o"].reshape(H, W, 3)
            rays_d = rays["rays_d"].reshape(H, W, 3)

            # Sample batch from interest region
            if interest_idxs.shape[0] <= self.batch_size:
                batch_idxs = interest_idxs
            else:
                idx = np.random.choice(interest_idxs.shape[0], self.batch_size, replace=False)
                batch_idxs = interest_idxs[idx]

            batch_y, batch_x = batch_idxs[:, 0], batch_idxs[:, 1]
            batch_y_t = torch.tensor(batch_y, dtype=torch.long, device='cuda')
            batch_x_t = torch.tensor(batch_x, dtype=torch.long, device='cuda')

            # Get rays for the sampled batch
            rays_o_batch = rays_o[batch_y_t, batch_x_t].unsqueeze(0)
            rays_d_batch = rays_d[batch_y_t, batch_x_t].unsqueeze(0)

            # Render the image from the current pose
            output = self.render_fn(rays_o_batch, rays_d_batch)

            # Important: Match the shape exactly for loss calculation
            rendered_rgb = output["image"].reshape(-1, 3) # Shape: [N, 3]

            # Get camera RGB values at the same pixels - ensure proper shape
            camera_rgb = camera_image_t[0, :, batch_y_t, batch_x_t].permute(1, 0) # Shape: [N, 3]

            # Make sure camera_rgb has gradients to avoid backward() error
            camera_rgb = camera_rgb.detach()

            # Ensure proper shapes before computing loss
            # print(f"rendered_rgb shape: {rendered_rgb.shape}, camera_rgb shape: {camera_rgb.shape}")

            # Use MSE loss with explicit shape matching
            loss = torch.nn.functional.mse_loss(rendered_rgb, camera_rgb)

            # Add small regularization term -- Weight Decay 
            reg_factor = 0.001
            reg_loss = reg_factor * torch.sum(pose_params**2)
            total_loss = loss + reg_loss

            # Backpropagate
            total_loss.backward()

            # # Print gradients before stepping the optimizer
            # print(f"Gradients for pose_params: {pose_params.grad}")

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Store loss
            losses.append(loss.item())

            # Print Progress -- Dynamically Update Print
            if iter <= 3 or iter % 50 == 0:
                print(f"Iteration {iter}, Loss: {loss.item()}")
                # print(f"Current params: {pose_params.data}")

            if self.render_viz and (iter <= 3 or iter % 100 == 0):
                # Render full image and save to render_viz folder for visualization
                with torch.no_grad():
                    full_rays = self.get_rays(pose_matrix)
                    full_output = self.render_fn(full_rays["rays_o"], full_rays["rays_d"])
                    full_rgb = full_output["image"].reshape(H, W, 3).cpu().numpy()
                
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(camera_image)
                plt.title("Camera Image")
                
                plt.subplot(1, 2, 2)
                plt.imshow(full_rgb)
                plt.title(f"Rendered at iter {iter}")
                
                plt.tight_layout()

                # Save to file
                viz_path = os.path.join(render_dir, f'localization_iter_{iter}.png')
                plt.savefig(viz_path)
                print(f"Visualization saved to {viz_path}")
                plt.close()


        # Extract final values
        x, y, z, yaw = [p.iten() for p in pose_params]

        # Create final pose matrix
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        
        final_pose = np.eye(4)
        final_pose[0, 0] = cos_y
        final_pose[0, 1] = -sin_y
        final_pose[1, 0] = sin_y
        final_pose[1, 1] = cos_y
        final_pose[0, 3] = x
        final_pose[1, 3] = y
        final_pose[2, 3] = z

        return final_pose[:3, 3], yaw, losses
    

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
                                            n_iters=1000, render_viz=True)
        
        result = optimizer.estimate_pose(camera_image) # Run Pose Optimizer on Image
        
        if result is not None:
            final_translation, final_yaw, losses = result
            # final_translation is a tuple (x, y, z)
            print("Final translation (x, y, z):", final_translation)
            print("Final yaw (radians):", final_yaw)

      
################## TEST ##################
# Load your sensor image (ensure it is in RGB).
camera_image = cv2.imread("1.png")

# camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

localize().run(camera_image)

            
