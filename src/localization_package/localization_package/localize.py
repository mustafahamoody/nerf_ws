import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# NeRF Functions
from nerf.utils import *
from nerf.provider import NeRFDataset
from nerf.network import NeRFNetwork
from occupancy_package.config.model_options import ModelOptions

# Helper: convert yaw (rotation about z) into a 3x3 rotation matrix.
def yaw_to_matrix(yaw):
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    R = torch.tensor([[cos_y, -sin_y, 0],
                      [sin_y, cos_y, 0],
                      [0,      0,     1]], dtype=yaw.dtype, device=yaw.device)
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
    def __init__(self, render_fn, get_rays_fn, learning_rate=0.01, n_itters=500, batch_size=1024, 
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
        self.n_itters = n_itters
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
            keypoints = find_keypoints(camera_image, render=self.render_viz)
            if keypoints.shape[0] == 0:
                print("No Features Found in Image")
                return None
        

            # Create mask from keypoints and dilate it
            intrest_mask = np.zero((H, W), dtype=np.uint8)
            # Keypoints are (x,y); use y for row and x for column.
            intrest_mask[keypoints[:, 1], keypoints[:, 0]] = 1 
            
            intrest_mask = cv2.dilate(intrest_mask, np.ones((self.kernal_size, self.kernal_size), np.uint8),
                                      iterations=self.dilate_iter)
            intrest_idxs = np.argwhere(intrest_mask > 0)


            # Initlaize x, y and yaw params -- As torch tenors with Grad. Used for pose optimization
            pose_params = torch.zeros(2, dtype=torch.float32, device='cuda', requires_grad=True)
            optimizer = torch.optim.Adam([pose_params], lr=self.learning_rate)
            losses = []


            # Optimization Loop
            for itter in range(1, self.n_itters + 1):

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
                        full_render = output_full["image"].reshape(H, W, 3).detatch().cpu().numpy()
                        plt.figure(figsize=(8, 8))
                        plt.imshow(full_render)
                        plt.title(f'Itteration {itter}')
                        plt.show()
                        plt.close()

                final_pose = pose_params.detach().cpu().numpy()
                final_translation = (final_pose[0], final_pose[1], final_pose[2])
                final_yaw = final_pose[3]
                return final_translation, final_yaw, losses


########################### NERF SETUP ###########################

# Initialize the NeRFNetwork model
model = NeRFNetwork(
    encoding=self.config_model['model']['encoding'],
    bound=self.config_model['model']['bound'],
    cuda_ray=self.config_model['model']['cuda_ray'],
    density_scale=self.config_model['model']['density_scale'],
    min_near=self.config_model['model']['min_near'],
    density_thresh=self.config_model['model']['density_thresh'],
    bg_radius=self.config_model['model']['bg_radius'],
)
model.eval()  # Set the model to evaluation mode

# Initialize the Trainer (this loads weights from a checkpoint)
trainer = Trainer(
    'ngp',
    opt=ModelOptions.opt(),
    model=self.model,
    device=self.device,
    workspace=self.config_trainer['trainer']['workspace'],
    criterion=self.criterion,
    fp16=self.config_model['model']['fp16'],
    use_checkpoint=self.config_trainer['trainer']['use_checkpoint'],
)


model.eval()
metrics = [PSNRMeter(),]
criterion = torch.nn.MSELoss(reduction='none')


dataset = NeRFDataset(ModelOptions.opt(), device=device, type='test')    
########################### NERF SETUP END ###########################

if __name__ == '__main__':

# ------------------------Example usage------------------------


    render_fn = lambda rays_o, rays_d: model.render(rays_o, rays_d, staged=True, bg_color=1., perturb=False, **vars(opt))
    get_rays_fn = lambda pose: get_rays(pose, dataset.intrinsics, dataset.H, dataset.W)

    # Load your sensor image (ensure it is in RGB).
    sensor_image = cv2.imread("path_to_image.png")
    sensor_image = cv2.cvtColor(sensor_image, cv2.COLOR_BGR2RGB)
    
    optimizer = PoseOptimizer(render_fn, get_rays_fn, lrate=0.005,
                                           num_iters=500, render_viz=True)
    result = optimizer.estimate_pose(sensor_image)
    if result is not None:
        final_translation, final_yaw, losses = result
        # final_translation is a tuple (x, y, z)
        print("Final translation (x, y, z):", final_translation)
        print("Final yaw (radians):", final_yaw)
