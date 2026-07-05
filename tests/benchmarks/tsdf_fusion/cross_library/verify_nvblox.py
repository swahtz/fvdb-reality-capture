import torch
import nvblox_torch
from nvblox_torch.mapper import Mapper
from nvblox_torch.sensor import Sensor
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams

print("=== (a) imports OK ===")
print("nvblox_torch", nvblox_torch.__version__, "| torch", torch.__version__,
      "| cuda", torch.version.cuda, "| device", torch.cuda.get_device_name(0))

print("\n=== (c) MapperParams introspection ===")
mp = MapperParams()
print("MapperParams dir():", [n for n in dir(mp) if not n.startswith("__")])
pip = mp.get_projective_integrator_params()
names = [n for n in dir(pip) if not n.startswith("_") and n != "wrap_getter_and_setters"]
print("\nProjectiveIntegratorParams attributes and defaults:")
for n in sorted(names):
    try:
        print(f"  {n} = {getattr(pip, n)}")
    except Exception as e:
        print(f"  {n} = <error: {e}>")

lidar_max = pip.lidar_projective_integrator_max_integration_distance_m
cam_max = pip.projective_integrator_max_integration_distance_m
trunc = pip.projective_integrator_truncation_distance_vox
print(f"\nKEY DEFAULTS: lidar_projective_integrator_max_integration_distance_m={lidar_max}")
print(f"KEY DEFAULTS: projective_integrator_max_integration_distance_m={cam_max}")
print(f"KEY DEFAULTS: projective_integrator_truncation_distance_vox={trunc}")

print("\n=== (b) Mapper + LiDAR TSDF integration on GPU ===")
mapper = Mapper(voxel_sizes_m=[0.2], integrator_types=[ProjectiveIntegratorType.TSDF])
sensor = Sensor.from_lidar(1800, 64, 0.4712, 1.0)
print("mapper:", mapper, "| sensor:", sensor)
depth = (torch.rand(64, 1800, device="cuda", dtype=torch.float32) * 45.0) + 5.0
pose = torch.eye(4, dtype=torch.float32)  # CPU per API docs
mapper.add_depth_frame(depth_frame=depth, t_w_c=pose, sensor=sensor)
torch.cuda.synchronize()
n_blocks = None
try:
    layer = mapper.tsdf_layer_view()
    n_blocks = layer.num_allocated_blocks()
except Exception as e:
    n_blocks = f"(introspection unavailable: {e})"
print("add_depth_frame integrated one 64x1800 LiDAR frame OK; tsdf blocks:", n_blocks)

print("\n=== (d) Mapper with non-default params (60.0 m, 3.0 vox) ===")
mp2 = MapperParams()
pip2 = mp2.get_projective_integrator_params()
pip2.lidar_projective_integrator_max_integration_distance_m = 60.0
pip2.projective_integrator_truncation_distance_vox = 3.0
mp2.set_projective_integrator_params(pip2)
check = mp2.get_projective_integrator_params()
print("after set: lidar max dist =", check.lidar_projective_integrator_max_integration_distance_m,
      "| trunc vox =", check.projective_integrator_truncation_distance_vox)
mapper2 = Mapper(voxel_sizes_m=[0.2], integrator_types=[ProjectiveIntegratorType.TSDF],
                 mapper_parameters=mp2)
depth2 = (torch.rand(64, 1800, device="cuda", dtype=torch.float32) * 45.0) + 5.0
mapper2.add_depth_frame(depth_frame=depth2, t_w_c=pose, sensor=sensor)
torch.cuda.synchronize()
print("Mapper with custom params constructed and integrated OK:", mapper2)

print("\nALL CHECKS PASSED")
