import sys
import os
import glob
import time
import math
import numpy as np
import cv2
import collections

# =========================================================
# 🔧 PATH FIX
# =========================================================
CARLA_ROOT = r"E:\CAR-simulator\CARLA_0.9.10\WindowsNoEditor"
API_PATH = os.path.join(CARLA_ROOT, "PythonAPI")
CARLA_MODULE_PATH = os.path.join(API_PATH, "carla") 

try:
    egg_path = glob.glob(os.path.join(CARLA_MODULE_PATH, "dist", "carla-*%d.%d-%s.egg" % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0]
    sys.path.append(egg_path)
except IndexError:
    print("⚠️ Warning: CARLA Egg file not found.")

if CARLA_MODULE_PATH not in sys.path:
    sys.path.append(CARLA_MODULE_PATH)

import carla
try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
except ImportError:
    print("❌ FATAL: Could not import 'agents'. Check path configuration.")
    sys.exit()

import torch
import torch.nn as nn

# =========================================================
# ⚙️ CONFIG
# =========================================================
MODEL_PATH = r"E:\CAR-simulator\models\kaggle\converted_model.pth"
CAM_W, CAM_H = 640, 480
CNN_W, CNN_H = 200, 66

# Driving Params
TARGET_SPEED = 30
OBSTACLE_SPEED = 10
STEER_GAIN = 1.0

# Map Window
MAP_WIN_W, MAP_WIN_H = 800, 800

# =========================================================
# 🗺️ DYNAMIC MAP ENGINE (Zoom/Pan)
# =========================================================
class MapEngine:
    def __init__(self, world):
        self.world = world
        self.map_w = MAP_WIN_W
        self.map_h = MAP_WIN_H
        
        # 1. Extract All Road Waypoints (High Res)
        print("🗺️  Caching World Map... (This may take 2s)")
        carla_map = world.get_map()
        self.waypoints = carla_map.generate_waypoints(2.0)
        
        # 2. Calculate World Bounds
        xs = [w.transform.location.x for w in self.waypoints]
        ys = [w.transform.location.y for w in self.waypoints]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)
        
        # Center of the world
        self.world_cx = (self.min_x + self.max_x) / 2
        self.world_cy = (self.min_y + self.max_y) / 2
        
        # 3. Auto-Fit Scale
        world_width = self.max_x - self.min_x
        world_height = self.max_y - self.min_y
        scale_x = (self.map_w - 100) / world_width
        scale_y = (self.map_h - 100) / world_height
        self.base_scale = min(scale_x, scale_y)
        
        # Current View State
        self.scale = self.base_scale
        self.offset_x = self.world_cx
        self.offset_y = self.world_cy
        self.is_dragging = False
        self.last_mouse = (0,0)

    def reset_view(self):
        self.scale = self.base_scale
        self.offset_x = self.world_cx
        self.offset_y = self.world_cy

    def world_to_screen(self, loc):
        """Converts World (X,Y) to Screen Pixels (U,V) based on Zoom/Pan"""
        # 1. Center the point relative to our offset
        x = loc.x - self.offset_x
        y = loc.y - self.offset_y
        
        # 2. Scale and move to center of screen
        u = int((x * self.scale) + (self.map_w / 2))
        v = int((y * self.scale) + (self.map_h / 2))
        
        # 3. Flip Y because computer screens start from top-left
        v = self.map_h - v 
        return (u, v)

    def screen_to_world(self, u, v):
        """Converts Mouse Click (U,V) back to World Location (X,Y)"""
        # Reverse Step 3
        v = self.map_h - v
        
        # Reverse Step 2
        x = (u - (self.map_w / 2)) / self.scale
        y = (v - (self.map_h / 2)) / self.scale
        
        # Reverse Step 1
        world_x = x + self.offset_x
        world_y = y + self.offset_y
        
        return carla.Location(x=world_x, y=world_y, z=0)

    def render(self, vehicle, route, start_pt, end_pt):
        # Create Black Canvas
        canvas = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
        
        # 1. Draw All Roads (Simple Gray Dots)
        # Optimization: Only draw points within view? 
        # For <10k points, numpy is fast enough to draw all.
        for w in self.waypoints:
            pt = self.world_to_screen(w.transform.location)
            # Clip check
            if 0 <= pt[0] < self.map_w and 0 <= pt[1] < self.map_h:
                canvas[pt[1], pt[0]] = (50, 50, 50) # BGR Gray

        # 2. Draw Route (Green Line)
        if route:
            for i in range(len(route) - 1):
                p1 = self.world_to_screen(route[i][0].transform.location)
                p2 = self.world_to_screen(route[i+1][0].transform.location)
                cv2.line(canvas, p1, p2, (0, 255, 0), 2)

        # 3. Draw Points
        if start_pt:
            p = self.world_to_screen(start_pt)
            cv2.circle(canvas, p, 6, (0, 200, 0), -1)
            cv2.putText(canvas, "START", (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        if end_pt:
            p = self.world_to_screen(end_pt)
            cv2.circle(canvas, p, 6, (0, 0, 255), -1)
            cv2.putText(canvas, "END", (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # 4. Draw Car (Yellow with heading)
        if vehicle:
            v_loc = vehicle.get_location()
            center = self.world_to_screen(v_loc)
            cv2.circle(canvas, center, 5, (0, 255, 255), -1)
            
            # Heading Arrow
            yaw = math.radians(vehicle.get_transform().rotation.yaw)
            arrow_len = 15
            end_x = center[0] + arrow_len * math.cos(yaw)
            end_y = center[1] - arrow_len * math.sin(yaw) # Y flip
            cv2.line(canvas, center, (int(end_x), int(end_y)), (0, 255, 255), 2)

        # UI Info
        cv2.putText(canvas, f"Zoom: {self.scale:.1f}x", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        return canvas

# Global Map Instance
MAP_ENGINE = None
MISSION_START = None
MISSION_END = None

def mouse_callback(event, x, y, flags, param):
    global MAP_ENGINE, MISSION_START, MISSION_END
    
    # --- ZOOM (Scroll) ---
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0: # Scroll Up
            MAP_ENGINE.scale *= 1.1
        else: # Scroll Down
            MAP_ENGINE.scale *= 0.9
        MAP_ENGINE.scale = max(0.5, min(MAP_ENGINE.scale, 20.0)) # Limits

    # --- PAN (Middle Click Drag) ---
    if event == cv2.EVENT_MBUTTONDOWN:
        MAP_ENGINE.is_dragging = True
        MAP_ENGINE.last_mouse = (x, y)
    elif event == cv2.EVENT_MBUTTONUP:
        MAP_ENGINE.is_dragging = False
    elif event == cv2.EVENT_MOUSEMOVE and MAP_ENGINE.is_dragging:
        dx = x - MAP_ENGINE.last_mouse[0]
        dy = y - MAP_ENGINE.last_mouse[1]
        # Adjust world offset (Opposite direction of drag)
        MAP_ENGINE.offset_x -= dx / MAP_ENGINE.scale
        MAP_ENGINE.offset_y += dy / MAP_ENGINE.scale # Y is inverted logic
        MAP_ENGINE.last_mouse = (x, y)

    # --- SET POINTS ---
    world_click = MAP_ENGINE.screen_to_world(x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        MISSION_START = world_click
        print(f"🟢 START: ({int(world_click.x)}, {int(world_click.y)})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        MISSION_END = world_click
        print(f"🔴 END:   ({int(world_click.x)}, {int(world_click.y)})")

# =========================================================
# MODELS & CONTROLLERS (Standard)
# =========================================================
class NvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1152, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 3)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.flat(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

class PID:
    def __init__(self):
        self.prev_err = 0
        self.integral = 0
    def run(self, target, current):
        err = target - current
        self.integral += err * 0.05
        deriv = (err - self.prev_err) / 0.05
        self.prev_err = err
        return np.clip(0.8 * err + 0.05 * self.integral + 0.05 * deriv, 0, 1)

def get_path_steer(vehicle, route):
    if not route: return 0.0, False
    loc = vehicle.get_location()
    yaw = vehicle.get_transform().rotation.yaw
    idx = min(4, len(route) - 1)
    target = route[idx][0].transform.location
    angle = math.degrees(math.atan2(target.y - loc.y, target.x - loc.x))
    diff = (angle - yaw + 180) % 360 - 180
    return np.clip(diff / 45.0, -1.0, 1.0), loc.distance(route[-1][0].transform.location) < 3.0

class CollisionSensor:
    def __init__(self, world, vehicle):
        self.sensor = world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),
                                        carla.Transform(), attach_to=vehicle)
        self.collision_flag = False
        self.sensor.listen(lambda event: self._on_collision(event))
    def _on_collision(self, event): self.collision_flag = True
    def check_and_reset(self):
        if self.collision_flag:
            self.collision_flag = False
            return True
        return False
    def destroy(self): self.sensor.destroy()

def process_lidar(lidar_data):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    lidar_points = points[:, :3]
    roi = np.where((lidar_points[:, 0] > 2.0) & (lidar_points[:, 0] < 10.0) & 
                   (lidar_points[:, 1] > -1.5) & (lidar_points[:, 1] < 1.5) & 
                   (lidar_points[:, 2] > -1.8))
    roi_pts = lidar_points[roi]
    if len(roi_pts) > 15: 
        obs_y = np.mean(roi_pts[:, 1])
        avoid_steer = 0.5 if obs_y < 0 else -0.5
        return avoid_steer, True
    return 0.0, False

def preprocess_rgb(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
    arr = arr[100:380] 
    arr = cv2.resize(arr, (CNN_W, CNN_H)) 
    arr = arr.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr).unsqueeze(0)

def get_display_img(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
    return arr.copy() 

def semantic_check(image):
    image.convert(carla.ColorConverter.CityScapesPalette)
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    raw = raw.reshape((image.height, image.width, 4))
    img = raw[:, :, :3]
    lower = np.array([110, 50, 110])
    upper = np.array([150, 80, 150])
    mask = cv2.inRange(img, lower, upper)
    roi = mask[int(CAM_H * 0.5):, :]
    return cv2.countNonZero(roi) / (roi.shape[0]*roi.shape[1])

# =========================================================
# 🚀 MAIN
# =========================================================
def main():
    global MAP_ENGINE
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")

    model = NvidiaModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()

    # Setup Route Planner
    dao = GlobalRoutePlannerDAO(carla_map, 2.0)
    grp = GlobalRoutePlanner(dao)
    grp.setup()

    # Setup Map Engine
    MAP_ENGINE = MapEngine(world)
    cv2.namedWindow("Mission Control", cv2.WINDOW_NORMAL) # Resizable
    cv2.resizeWindow("Mission Control", 800, 800)
    cv2.setMouseCallback("Mission Control", mouse_callback)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    vehicle = world.spawn_actor(bp_lib.filter("model3")[0], carla_map.get_spawn_points()[0])
    
    # Sensors
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAM_W))
    cam_bp.set_attribute("image_size_y", str(CAM_H))
    
    sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")
    sem_bp.set_attribute("image_size_x", str(CAM_W))
    sem_bp.set_attribute("image_size_y", str(CAM_H))

    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute('range', '20')
    lidar_bp.set_attribute('rotation_frequency', '20')

    camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
    rear_cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=180)), attach_to=vehicle)
    sem_cam = world.spawn_actor(sem_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
    col_sensor = CollisionSensor(world, vehicle)

    q_cam = collections.deque(maxlen=1)
    q_lidar = collections.deque(maxlen=1)
    q_rear = collections.deque(maxlen=1)
    q_sem = collections.deque(maxlen=1)
    
    camera.listen(q_cam.append)
    lidar.listen(q_lidar.append)
    rear_cam.listen(q_rear.append)
    sem_cam.listen(q_sem.append)

    pid = PID()
    route = []
    state = "IDLE"
    recovery_timer = 0
    finish_timer = 0
    extra_reverse = False
    
    print("\n✅ READY. Controls:")
    print("   [Scroll] Zoom | [Middle Click] Pan | [R] Reset Map")
    print("   [L-Click] Start | [R-Click] End")

    try:
        while True:
            world.tick()
            if not q_cam or not q_lidar or not q_rear or not q_sem: continue
            
            img_data = q_cam.pop()
            lidar_data = q_lidar.pop()
            rear_data = q_rear.pop()
            sem_data = q_sem.pop()

            # --- MISSION PLANNER ---
            if state == "IDLE":
                if MISSION_START and MISSION_END:
                    state = "CALC_TO_START"

            elif state == "CALC_TO_START":
                print("🔄 Path to START...")
                curr_w = carla_map.get_waypoint(vehicle.get_location())
                start_w = carla_map.get_waypoint(MISSION_START)
                route = grp.trace_route(curr_w.transform.location, start_w.transform.location)
                state = "TO_START"

            elif state == "CALC_TO_END":
                print("🔄 Path to END...")
                start_w = carla_map.get_waypoint(vehicle.get_location())
                end_w = carla_map.get_waypoint(MISSION_END)
                route = grp.trace_route(start_w.transform.location, end_w.transform.location)
                state = "TO_END"

            # --- CONTROL LOOP ---
            th, st, br = 0.0, 0.0, 1.0
            
            if col_sensor.check_and_reset():
                state = "RECOVERING"
                recovery_timer = time.time() + 4.0
                extra_reverse = False
                print("💥 CRASH!")

            if state == "RECOVERING":
                road_conf = semantic_check(sem_data)
                if road_conf > 0.3 and not extra_reverse:
                    recovery_timer = time.time() + 2.0
                    extra_reverse = True
                
                if time.time() > recovery_timer:
                    if MISSION_START and MISSION_END:
                        state = "CALC_TO_END" 
                    else:
                        state = "IDLE"
                else:
                    th, st, br = 0.5, 0.0, 0.0
                    vehicle.apply_control(carla.VehicleControl(throttle=th, steer=st, brake=br, reverse=True))
                    r_img = get_display_img(rear_data)
                    cv2.putText(r_img, "REVERSING...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.imshow("View", r_img)

            elif state in ["TO_START", "TO_END"]:
                my_loc = vehicle.get_location()
                min_d = 1000
                idx = 0
                for i in range(min(15, len(route))):
                    d = my_loc.distance(route[i][0].transform.location)
                    if d < min_d: min_d, idx = d, i
                route = route[idx:]

                path_steer, arrived = get_path_steer(vehicle, route)
                avoid_steer, obstacle = process_lidar(lidar_data)

                if arrived:
                    if state == "TO_START":
                        print("✅ Reached Start! Proceeding to End...")
                        state = "CALC_TO_END"
                    elif state == "TO_END":
                        print("🏆 Mission Complete!")
                        state = "FINISHED"
                        finish_timer = time.time() + 3.0
                else:
                    if obstacle:
                        st = avoid_steer
                        target_s = OBSTACLE_SPEED
                        status = "⚠️ AVOID"
                    else:
                        cnn_out = model(preprocess_rgb(img_data).to(device))[0].detach().cpu().numpy()
                        cnn_st = float(cnn_out[0]) * STEER_GAIN
                        st = 0.4 * path_steer + 0.6 * cnn_st
                        target_s = TARGET_SPEED if abs(st) < 0.2 else 15
                        status = "MISSION"

                    v = 3.6 * math.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)
                    th = pid.run(target_s, v)
                    br = 0.0 if v < target_s + 5 else 0.5
                    vehicle.apply_control(carla.VehicleControl(throttle=th, steer=float(st), brake=br))

                    f_img = get_display_img(img_data)
                    cv2.putText(f_img, f"MODE: {status} | STAGE: {state}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.imshow("View", f_img)

            elif state == "FINISHED":
                vehicle.apply_control(carla.VehicleControl(hand_brake=True))
                f_img = get_display_img(img_data)
                remaining = int(finish_timer - time.time())
                cv2.putText(f_img, f"EXITING IN {remaining}...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.imshow("View", f_img)
                if time.time() > finish_timer: break

            else: # IDLE
                vehicle.apply_control(carla.VehicleControl(hand_brake=True))
                f_img = get_display_img(img_data)
                cv2.putText(f_img, "WAITING FOR POINTS...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.imshow("View", f_img)

            # RENDER DYNAMIC MAP
            map_view = MAP_ENGINE.render(vehicle, route, MISSION_START, MISSION_END)
            cv2.imshow("Mission Control", map_view)

            key = cv2.waitKey(1)
            if key == ord('q'): break
            if key == ord('r'): MAP_ENGINE.reset_view()

    finally:
        vehicle.destroy()
        camera.destroy()
        lidar.destroy()
        rear_cam.destroy()
        col_sensor.destroy()
        sem_cam.destroy()
        cv2.destroyAllWindows()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Clean Exit.")

if __name__ == "__main__":
    main()