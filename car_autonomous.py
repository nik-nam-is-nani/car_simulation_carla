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
TARGET_SPEED    = 20
OBSTACLE_SPEED  = 6
FOLLOW_SPEED    = 12
STEER_GAIN      = 1.0

# LiDAR distance zones (metres)
LIDAR_STOP_DIST   = 3.5
LIDAR_SLOW_DIST   = 12.0
LIDAR_FOLLOW_DIST = 6.0

# LiDAR lane geometry
LIDAR_LANE_HALF_W = 1.2
LIDAR_MIN_HEIGHT  = -1.5
LIDAR_MAX_HEIGHT  =  1.0
LIDAR_MIN_X       =  2.5

# Pedestrian-specific LiDAR config
LIDAR_PED_MIN_Z   = -1.6
LIDAR_PED_MAX_Z   =  0.5

# Map Window
MAP_WIN_W, MAP_WIN_H = 800, 800

# =========================================================
# 🎥 CAMERA VIEWS
# Format: (x, y, z, pitch, yaw)
# =========================================================
CAMERA_VIEWS = [
    ( 1.5,  0.0, 2.4,   0,   0),   # 0 — Front (default)
    (-1.5,  0.0, 2.4,   0, 180),   # 1 — Rear
    ( 0.0, -1.2, 2.4,   0, -90),   # 2 — Left side
    ( 0.0,  1.2, 2.4,   0,  90),   # 3 — Right side
    ( 0.0,  0.0, 8.0, -70,   0),   # 4 — Top-down bird's eye
    (-4.0,  0.0, 3.5, -15, 180),   # 5 — Chase / cinematic rear
]
CAMERA_VIEW_NAMES = [
    "FRONT", "REAR", "LEFT", "RIGHT", "BIRD'S EYE", "CHASE"
]
current_view_idx = 0

# =========================================================
# 🗺️ DYNAMIC MAP ENGINE (Zoom/Pan)
# =========================================================
class MapEngine:
    def __init__(self, world):
        self.world = world
        self.map_w = MAP_WIN_W
        self.map_h = MAP_WIN_H

        print("🗺️  Caching World Map... (This may take 2s)")
        carla_map = world.get_map()
        self.waypoints = carla_map.generate_waypoints(2.0)

        xs = [w.transform.location.x for w in self.waypoints]
        ys = [w.transform.location.y for w in self.waypoints]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)

        self.world_cx = (self.min_x + self.max_x) / 2
        self.world_cy = (self.min_y + self.max_y) / 2

        world_width  = self.max_x - self.min_x
        world_height = self.max_y - self.min_y
        scale_x = (self.map_w - 100) / world_width
        scale_y = (self.map_h - 100) / world_height
        self.base_scale = min(scale_x, scale_y)

        self.scale    = self.base_scale
        self.offset_x = self.world_cx
        self.offset_y = self.world_cy
        self.is_dragging = False
        self.last_mouse  = (0, 0)

    def reset_view(self):
        self.scale    = self.base_scale
        self.offset_x = self.world_cx
        self.offset_y = self.world_cy

    def world_to_screen(self, loc):
        x = loc.x - self.offset_x
        y = loc.y - self.offset_y
        u = int((x * self.scale) + (self.map_w / 2))
        v = int((y * self.scale) + (self.map_h / 2))
        v = self.map_h - v
        return (u, v)

    def screen_to_world(self, u, v):
        v = self.map_h - v
        x = (u - (self.map_w / 2)) / self.scale
        y = (v - (self.map_h / 2)) / self.scale
        world_x = x + self.offset_x
        world_y = y + self.offset_y
        return carla.Location(x=world_x, y=world_y, z=0)

    def render(self, vehicle, route, start_pt, end_pt):
        canvas = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)

        for w in self.waypoints:
            pt = self.world_to_screen(w.transform.location)
            if 0 <= pt[0] < self.map_w and 0 <= pt[1] < self.map_h:
                canvas[pt[1], pt[0]] = (50, 50, 50)

        if route:
            for i in range(len(route) - 1):
                p1 = self.world_to_screen(route[i][0].transform.location)
                p2 = self.world_to_screen(route[i+1][0].transform.location)
                cv2.line(canvas, p1, p2, (0, 255, 0), 2)

        if start_pt:
            p = self.world_to_screen(start_pt)
            cv2.circle(canvas, p, 6, (0, 200, 0), -1)
            cv2.putText(canvas, "START", (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if end_pt:
            p = self.world_to_screen(end_pt)
            cv2.circle(canvas, p, 6, (0, 0, 255), -1)
            cv2.putText(canvas, "END", (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        if vehicle:
            v_loc  = vehicle.get_location()
            center = self.world_to_screen(v_loc)
            cv2.circle(canvas, center, 5, (0, 255, 255), -1)
            yaw      = math.radians(vehicle.get_transform().rotation.yaw)
            arrow_len = 15
            end_x = center[0] + arrow_len * math.cos(yaw)
            end_y = center[1] - arrow_len * math.sin(yaw)
            cv2.line(canvas, center, (int(end_x), int(end_y)), (0, 255, 255), 2)

        cv2.putText(canvas, f"Zoom: {self.scale:.1f}x", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        return canvas

# Global Map Instance
MAP_ENGINE    = None
MISSION_START = None
MISSION_END   = None

def mouse_callback(event, x, y, flags, param):
    global MAP_ENGINE, MISSION_START, MISSION_END

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            MAP_ENGINE.scale *= 1.1
        else:
            MAP_ENGINE.scale *= 0.9
        MAP_ENGINE.scale = max(0.5, min(MAP_ENGINE.scale, 20.0))

    if event == cv2.EVENT_MBUTTONDOWN:
        MAP_ENGINE.is_dragging = True
        MAP_ENGINE.last_mouse  = (x, y)
    elif event == cv2.EVENT_MBUTTONUP:
        MAP_ENGINE.is_dragging = False
    elif event == cv2.EVENT_MOUSEMOVE and MAP_ENGINE.is_dragging:
        dx = x - MAP_ENGINE.last_mouse[0]
        dy = y - MAP_ENGINE.last_mouse[1]
        MAP_ENGINE.offset_x -= dx / MAP_ENGINE.scale
        MAP_ENGINE.offset_y += dy / MAP_ENGINE.scale
        MAP_ENGINE.last_mouse = (x, y)

    world_click = MAP_ENGINE.screen_to_world(x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        MISSION_START = world_click
        print(f"🟢 START: ({int(world_click.x)}, {int(world_click.y)})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        MISSION_END = world_click
        print(f"🔴 END:   ({int(world_click.x)}, {int(world_click.y)})")

# =========================================================
# MODELS & CONTROLLERS
# =========================================================
class NvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.fc1   = nn.Linear(1152, 100)
        self.fc2   = nn.Linear(100, 50)
        self.fc3   = nn.Linear(50, 10)
        self.fc4   = nn.Linear(10, 3)
        self.relu  = nn.ReLU()
        self.flat  = nn.Flatten()

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
    idx    = min(4, len(route) - 1)
    target = route[idx][0].transform.location
    angle  = math.degrees(math.atan2(target.y - loc.y, target.x - loc.x))
    diff   = (angle - yaw + 180) % 360 - 180
    return np.clip(diff / 45.0, -1.0, 1.0), loc.distance(route[-1][0].transform.location) < 3.0

class CollisionSensor:
    def __init__(self, world, vehicle):
        self.sensor = world.spawn_actor(
            world.get_blueprint_library().find('sensor.other.collision'),
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

# =========================================================
# 🚦 TRAFFIC LIGHT HELPER
# =========================================================
def check_traffic_light(vehicle):
    if vehicle.is_at_traffic_light():
        tl    = vehicle.get_traffic_light()
        state = tl.get_state()

        v_loc      = vehicle.get_location()
        tl_loc     = tl.get_location()
        v_yaw      = math.radians(vehicle.get_transform().rotation.yaw)
        fwd_x, fwd_y = math.cos(v_yaw), math.sin(v_yaw)
        dx = tl_loc.x - v_loc.x
        dy = tl_loc.y - v_loc.y
        dot = fwd_x * dx + fwd_y * dy

        if dot <= 0:
            return False, "NONE"

        if state == carla.TrafficLightState.Red:
            return True, "RED"
        elif state == carla.TrafficLightState.Yellow:
            return True, "YELLOW"
        else:
            return False, "GREEN"
    return False, "NONE"

# =========================================================
# 🆕 ADVANCED LiDAR PROCESSOR
# =========================================================
def process_lidar(lidar_data):
    points    = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points    = np.reshape(points, (int(points.shape[0] / 4), 4))
    lidar_pts = points[:, :3]

    front_dist   = LIDAR_SLOW_DIST + 1.0
    avoid_steer  = 0.0
    obstacle     = False
    lidar_status = "CLEAR"

    # Zone A: Vehicles / bikes in lane
    mask_a = (
        (lidar_pts[:, 0] > LIDAR_MIN_X) &
        (lidar_pts[:, 0] < LIDAR_SLOW_DIST) &
        (lidar_pts[:, 1] > -LIDAR_LANE_HALF_W) &
        (lidar_pts[:, 1] <  LIDAR_LANE_HALF_W) &
        (lidar_pts[:, 2] > LIDAR_MIN_HEIGHT) &
        (lidar_pts[:, 2] <  1.6)
    )
    zone_a_pts = lidar_pts[mask_a]

    if len(zone_a_pts) >= 5:
        front_dist = float(np.min(zone_a_pts[:, 0]))
        obs_y      = float(np.mean(zone_a_pts[:, 1]))

        if front_dist < LIDAR_STOP_DIST:
            obstacle     = True
            lidar_status = "STOP"
        elif front_dist < LIDAR_FOLLOW_DIST:
            obstacle     = True
            avoid_steer  = 0.3 if obs_y < 0 else -0.3
            lidar_status = "FOLLOW"
        else:
            obstacle     = True
            avoid_steer  = 0.5 if obs_y < 0 else -0.5
            lidar_status = "SLOW"

    # Zone B: Pedestrian detection
    if lidar_status == "CLEAR":
        mask_ped = (
            (lidar_pts[:, 0] > LIDAR_MIN_X) &
            (lidar_pts[:, 0] < 15.0) &
            (lidar_pts[:, 1] > -1.5) &
            (lidar_pts[:, 1] <  1.5) &
            (lidar_pts[:, 2] > LIDAR_PED_MIN_Z) &
            (lidar_pts[:, 2] <  LIDAR_PED_MAX_Z)
        )
        ped_pts = lidar_pts[mask_ped]

        if len(ped_pts) >= 3:
            z_spread = float(np.max(ped_pts[:, 2]) - np.min(ped_pts[:, 2]))
            if z_spread > 0.35:
                ped_dist = float(np.min(ped_pts[:, 0]))
                if ped_dist < front_dist:
                    front_dist  = ped_dist
                    obs_y       = float(np.mean(ped_pts[:, 1]))
                    avoid_steer = 0.4 if obs_y < 0 else -0.4
                    obstacle    = True
                    if ped_dist < LIDAR_STOP_DIST + 1.0:
                        lidar_status = "PED-STOP"
                    else:
                        lidar_status = "PED-SLOW"

    # Zone C: Crossing pedestrians / cyclists
    if lidar_status == "CLEAR":
        mask_c = (
            (lidar_pts[:, 0] > LIDAR_MIN_X) &
            (lidar_pts[:, 0] < 9.0) &
            (lidar_pts[:, 1] > -2.5) &
            (lidar_pts[:, 1] <  2.5) &
            (lidar_pts[:, 2] > LIDAR_PED_MIN_Z) &
            (lidar_pts[:, 2] <  1.6)
        )
        zone_c_pts = lidar_pts[mask_c]

        if len(zone_c_pts) >= 8:
            z_spread_c = float(np.max(zone_c_pts[:, 2]) - np.min(zone_c_pts[:, 2]))
            if z_spread_c > 0.3:
                cross_dist = float(np.min(zone_c_pts[:, 0]))
                if cross_dist < 8.0:
                    obstacle     = True
                    lidar_status = "CROSSING"
                    front_dist   = cross_dist

    return avoid_steer, obstacle, front_dist, lidar_status


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
    mask  = cv2.inRange(img, lower, upper)
    roi   = mask[int(CAM_H * 0.5):, :]
    return cv2.countNonZero(roi) / (roi.shape[0] * roi.shape[1])

# =========================================================
# 🚀 MAIN
# =========================================================
def main():
    global MAP_ENGINE, current_view_idx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")

    model = NvidiaModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    client    = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world     = client.get_world()
    carla_map = world.get_map()

    dao = GlobalRoutePlannerDAO(carla_map, 2.0)
    grp = GlobalRoutePlanner(dao)
    grp.setup()

    MAP_ENGINE = MapEngine(world)
    cv2.namedWindow("Mission Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mission Control", 800, 800)
    cv2.setMouseCallback("Mission Control", mouse_callback)

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib  = world.get_blueprint_library()
    vehicle = world.spawn_actor(bp_lib.filter("model3")[0], carla_map.get_spawn_points()[0])

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAM_W))
    cam_bp.set_attribute("image_size_y", str(CAM_H))

    sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")
    sem_bp.set_attribute("image_size_x", str(CAM_W))
    sem_bp.set_attribute("image_size_y", str(CAM_H))

    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute('range', '20')
    lidar_bp.set_attribute('rotation_frequency', '20')

    # 🎥 AI camera — fixed front, feeds CNN & path logic
    ai_cam = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        attach_to=vehicle
    )

    # 🎥 Display camera — switchable view
    v0 = CAMERA_VIEWS[current_view_idx]
    display_cam = world.spawn_actor(
        cam_bp,
        carla.Transform(
            carla.Location(x=v0[0], y=v0[1], z=v0[2]),
            carla.Rotation(pitch=v0[3], yaw=v0[4])
        ),
        attach_to=vehicle
    )

    # Rear cam kept for RECOVERING state display
    rear_cam  = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=180)),
        attach_to=vehicle
    )
    sem_cam   = world.spawn_actor(sem_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
    lidar     = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
    col_sensor = CollisionSensor(world, vehicle)

    q_display = collections.deque(maxlen=1)
    q_ai      = collections.deque(maxlen=1)
    q_rear    = collections.deque(maxlen=1)
    q_lidar   = collections.deque(maxlen=1)
    q_sem     = collections.deque(maxlen=1)

    display_cam.listen(q_display.append)
    ai_cam.listen(q_ai.append)
    rear_cam.listen(q_rear.append)
    lidar.listen(q_lidar.append)
    sem_cam.listen(q_sem.append)

    pid           = PID()
    route         = []
    state         = "IDLE"
    recovery_timer = 0
    finish_timer   = 0
    extra_reverse  = False

    realign_timer             = 0.0
    realign_steer             = 0.0
    prev_state_before_realign = "TO_END"

    print("\n✅ READY. Controls:")
    print("   [V] Cycle camera view | [Scroll] Zoom | [Middle Click] Pan | [R] Reset Map")
    print("   [L-Click] Start | [R-Click] End | [Q] Quit")
    print(f"   Camera views: {CAMERA_VIEW_NAMES}")

    try:
        while True:
            world.tick()
            if not q_display or not q_ai or not q_lidar or not q_rear or not q_sem:
                continue

            img_display = q_display.pop()
            img_ai      = q_ai.pop()
            lidar_data  = q_lidar.pop()
            rear_data   = q_rear.pop()
            sem_data    = q_sem.pop()

            # --- MISSION PLANNER ---
            if state == "IDLE":
                if MISSION_START and MISSION_END:
                    state = "CALC_TO_START"

            elif state == "CALC_TO_START":
                print("🔄 Path to START...")
                curr_w  = carla_map.get_waypoint(vehicle.get_location())
                start_w = carla_map.get_waypoint(MISSION_START)
                route   = grp.trace_route(curr_w.transform.location, start_w.transform.location)
                state   = "TO_START"

            elif state == "CALC_TO_END":
                print("🔄 Path to END...")
                start_w = carla_map.get_waypoint(vehicle.get_location())
                end_w   = carla_map.get_waypoint(MISSION_END)
                route   = grp.trace_route(start_w.transform.location, end_w.transform.location)
                state   = "TO_END"

            # --- CONTROL LOOP ---
            th, st, br = 0.0, 0.0, 1.0

            if col_sensor.check_and_reset():
                state          = "RECOVERING"
                recovery_timer = time.time() + 4.0
                extra_reverse  = False
                print("💥 CRASH!")

            if state == "RECOVERING":
                road_conf = semantic_check(sem_data)
                if road_conf > 0.3 and not extra_reverse:
                    recovery_timer = time.time() + 2.0
                    extra_reverse  = True

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

            elif state == "REALIGN":
                if time.time() > realign_timer:
                    state = "CALC_TO_END" if prev_state_before_realign == "TO_END" else "CALC_TO_START"
                    print("✅ Realign done — recalculating route")
                else:
                    remaining_t = realign_timer - time.time()
                    vehicle.apply_control(carla.VehicleControl(
                        throttle=0.35,
                        steer=float(realign_steer),
                        brake=0.0,
                        reverse=True
                    ))
                    r_img = get_display_img(rear_data)
                    cv2.putText(r_img, f"REALIGNING... {remaining_t:.1f}s",
                                (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                    cv2.imshow("View", r_img)

            elif state in ["TO_START", "TO_END"]:
                my_loc = vehicle.get_location()
                min_d  = 1000
                idx    = 0
                for i in range(min(15, len(route))):
                    d = my_loc.distance(route[i][0].transform.location)
                    if d < min_d: min_d, idx = d, i
                route = route[idx:]

                path_steer, arrived = get_path_steer(vehicle, route)
                avoid_steer, obstacle, front_dist, lidar_status = process_lidar(lidar_data)
                tl_stop, tl_color = check_traffic_light(vehicle)

                if arrived:
                    if state == "TO_START":
                        print("✅ Reached Start! Proceeding to End...")
                        state = "CALC_TO_END"
                    elif state == "TO_END":
                        print("🏆 Mission Complete!")
                        state        = "FINISHED"
                        finish_timer = time.time() + 3.0
                else:
                    # Priority 1: Traffic light
                    if tl_stop:
                        th, st, br = 0.0, path_steer * 0.3, 1.0
                        status = f"🚦 {tl_color} LIGHT"

                    # Priority 2: LiDAR hard-stop
                    elif lidar_status in ("STOP", "CROSSING", "PED-STOP"):
                        th, st, br = 0.0, path_steer * 0.3, 1.0
                        status = f"🛑 {lidar_status}"

                    # Priority 2b: Pedestrian slow-down
                    elif lidar_status == "PED-SLOW":
                        slow_factor = max(0.0, (front_dist - LIDAR_STOP_DIST) /
                                               (LIDAR_SLOW_DIST - LIDAR_STOP_DIST))
                        target_s    = 5.0 + (TARGET_SPEED - 5.0) * slow_factor
                        v           = 3.6 * math.sqrt(vehicle.get_velocity().x**2 +
                                                       vehicle.get_velocity().y**2)
                        th          = pid.run(target_s, v)
                        br          = 0.8 if v > target_s + 2 else 0.0
                        st          = 0.7 * path_steer + 0.3 * avoid_steer
                        status      = f"🚶 PED ({front_dist:.1f}m)"

                    # Priority 3: Following vehicle ahead
                    elif lidar_status == "FOLLOW":
                        gap_ratio  = max(0.0, (front_dist - LIDAR_STOP_DIST) /
                                               (LIDAR_FOLLOW_DIST - LIDAR_STOP_DIST))
                        target_s   = FOLLOW_SPEED * gap_ratio
                        v          = 3.6 * math.sqrt(vehicle.get_velocity().x**2 +
                                                      vehicle.get_velocity().y**2)
                        th         = pid.run(target_s, v)
                        br         = 0.5 if v > target_s + 3 else 0.0
                        st         = 0.4 * path_steer + 0.6 * avoid_steer
                        status     = f"🚗 FOLLOW ({front_dist:.1f}m)"

                    # Priority 4: Obstacle slow zone
                    elif obstacle:
                        slow_factor = (front_dist - LIDAR_STOP_DIST) / \
                                      (LIDAR_SLOW_DIST - LIDAR_STOP_DIST)
                        target_s    = OBSTACLE_SPEED + (TARGET_SPEED - OBSTACLE_SPEED) * slow_factor
                        v           = 3.6 * math.sqrt(vehicle.get_velocity().x**2 +
                                                       vehicle.get_velocity().y**2)
                        th          = pid.run(target_s, v)
                        br          = 0.0 if v < target_s + 5 else 0.5
                        st          = avoid_steer
                        status      = f"⚠️ SLOW ({front_dist:.1f}m)"

                    # Priority 5: Normal driving (CNN + path)
                    else:
                        # Use ai_cam (fixed front) for CNN inference — always correct angle
                        cnn_out  = model(preprocess_rgb(img_ai).to(device))[0].detach().cpu().numpy()
                        cnn_st   = float(cnn_out[0]) * STEER_GAIN
                        st       = 0.75 * path_steer + 0.25 * cnn_st
                        target_s = TARGET_SPEED if abs(st) < 0.2 else 15
                        v        = 3.6 * math.sqrt(vehicle.get_velocity().x**2 +
                                                    vehicle.get_velocity().y**2)
                        th       = pid.run(target_s, v)
                        br       = 0.0 if v < target_s + 5 else 0.5
                        status   = "MISSION"

                        # Lane-deviation check
                        wp_now = carla_map.get_waypoint(my_loc, project_to_road=True,
                                                        lane_type=carla.LaneType.Driving)
                        lane_offset = abs(wp_now.transform.location.x - my_loc.x) + \
                                      abs(wp_now.transform.location.y - my_loc.y)
                        if lane_offset > 2.5 and v < 8.0 and state in ["TO_START", "TO_END"]:
                            realign_steer             = -np.sign(path_steer) * 0.4
                            realign_timer             = time.time() + 1.5
                            prev_state_before_realign = state
                            state                     = "REALIGN"
                            print(f"↩️  Lane deviation {lane_offset:.1f}m — reversing to realign")

                    vehicle.apply_control(carla.VehicleControl(
                        throttle=float(th),
                        steer=float(st),
                        brake=float(br)
                    ))

                    # HUD overlay — rendered on display_cam (switchable view)
                    f_img = get_display_img(img_display)
                    v_kmh = 3.6 * math.sqrt(vehicle.get_velocity().x**2 +
                                             vehicle.get_velocity().y**2)
                    cv2.putText(f_img, f"MODE: {status}",
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(f_img, f"STAGE: {state}  |  {v_kmh:.1f} km/h",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    # Camera view label
                    cv2.putText(f_img, f"CAM: {CAMERA_VIEW_NAMES[current_view_idx]}  [V] to switch",
                                (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                    # Traffic light indicator
                    tl_color_bgr = {"RED":(0,0,255), "YELLOW":(0,200,255),
                                    "GREEN":(0,255,0), "NONE":(100,100,100)}
                    cv2.circle(f_img, (610, 30), 18,
                               tl_color_bgr.get(tl_color, (100,100,100)), -1)
                    cv2.putText(f_img, tl_color,
                                (575, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                    # LiDAR distance bar
                    bar_w = int(np.clip((front_dist / LIDAR_SLOW_DIST), 0, 1) * 200)
                    cv2.rectangle(f_img, (20, 80), (220, 95), (50,50,50), -1)
                    cv2.rectangle(f_img, (20, 80), (20+bar_w, 95), (0,200,255), -1)
                    cv2.putText(f_img, f"LIDAR: {front_dist:.1f}m",
                                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)
                    cv2.imshow("View", f_img)

            elif state == "FINISHED":
                vehicle.apply_control(carla.VehicleControl(hand_brake=True))
                f_img     = get_display_img(img_display)
                remaining = int(finish_timer - time.time())
                cv2.putText(f_img, f"EXITING IN {remaining}...",
                            (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.imshow("View", f_img)
                if time.time() > finish_timer: break

            else:  # IDLE
                vehicle.apply_control(carla.VehicleControl(hand_brake=True))
                f_img = get_display_img(img_display)
                cv2.putText(f_img, "WAITING FOR POINTS...",
                            (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(f_img, f"CAM: {CAMERA_VIEW_NAMES[current_view_idx]}  [V] to switch",
                            (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)
                cv2.imshow("View", f_img)

            # RENDER DYNAMIC MAP
            map_view = MAP_ENGINE.render(vehicle, route, MISSION_START, MISSION_END)
            cv2.imshow("Mission Control", map_view)

            key = cv2.waitKey(1)
            if key == ord('q'): break
            if key == ord('r'): MAP_ENGINE.reset_view()

            # 🎥 CAMERA VIEW CYCLE — press V
            if key == ord('v'):
                current_view_idx = (current_view_idx + 1) % len(CAMERA_VIEWS)
                cv = CAMERA_VIEWS[current_view_idx]
                display_cam.set_transform(carla.Transform(
                    carla.Location(x=cv[0], y=cv[1], z=cv[2]),
                    carla.Rotation(pitch=cv[3], yaw=cv[4])
                ))
                q_display.clear()  # flush stale frame for instant snap
                print(f"🎥 Camera → {CAMERA_VIEW_NAMES[current_view_idx]}")

    finally:
        vehicle.destroy()
        display_cam.destroy()
        ai_cam.destroy()
        rear_cam.destroy()
        lidar.destroy()
        col_sensor.destroy()
        sem_cam.destroy()
        cv2.destroyAllWindows()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Clean Exit.")

if __name__ == "__main__":
    main()
