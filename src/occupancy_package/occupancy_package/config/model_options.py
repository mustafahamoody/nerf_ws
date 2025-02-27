import argparse

# Change according to NeRF env values
bound = 2.0 # Default (Axis-ALigned) Bounding Box scale
scale = 0.5 # Default scale
dt_gamma = 0.0 # Default dt_gamma
density_thresh = 10.0 # Default density threshold
iters = 40000 # Default number of iterations

class ModelOptions:

    @staticmethod
    def opt():
        parser = argparse.ArgumentParser()
        # parser.add_argument('path', type=str)
        parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload", default=True)
        parser.add_argument('--test', action='store_true', help="test mode")
        # parser.add_argument('--workspace', type=str, default='workspace')
        parser.add_argument('--seed', type=int, default=0)

        ### training options
        parser.add_argument('--iters', type=int, default=iters, help="training iters")
        parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
        parser.add_argument('--ckpt', type=str, default='latest')
        parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
        parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
        parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
        parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
        parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
        parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
        parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
        parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

        ### network backbone options
        parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
        parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
        parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

        ### dataset options
        parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
        parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
        # (the default value is for the fox dataset)
        parser.add_argument('--bound', type=float, default=bound, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
        parser.add_argument('--scale', type=float, default=scale, help="scale camera location into box[-bound, bound]^3")
        parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
        parser.add_argument('--dt_gamma', type=float, default=dt_gamma, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
        parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
        parser.add_argument('--density_thresh', type=float, default=density_thresh, help="threshold for density grid to be occupied")
        parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

        ### GUI options
        parser.add_argument('--gui', action='store_true', help="start a GUI")
        parser.add_argument('--W', type=int, default=1920, help="GUI width")
        parser.add_argument('--H', type=int, default=1080, help="GUI height")
        parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
        parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
        parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

        ### experimental
        parser.add_argument('--error_map', action='store_true', help="use error map to sample rays", default=True)
        parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
        parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

        opt = parser.parse_args()
        return opt