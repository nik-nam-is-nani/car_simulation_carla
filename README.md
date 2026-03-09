

---

# 🚗 CARLA Hybrid Autonomous Driving System

> **Deep Learning + Classical Planning + Multi-Sensor Fusion**

A **hybrid autonomous driving system** built using **CARLA Simulator**, combining **Deep Learning Behavioral Cloning**, **Classical Path Planning**, and **Sensor Fusion** for realistic autonomous navigation.

This project demonstrates how **modern autonomous vehicles integrate learning-based and rule-based systems** to achieve robust driving behavior.

---

# 🌟 Features

✨ **End-to-End CNN Steering**
✨ **Global Route Planning** using CARLA navigation stack
✨ **LiDAR-based Obstacle Detection & Avoidance**
✨ **Traffic Light Detection & Handling**
✨ **Semantic Segmentation for Road Confidence**
✨ **Dynamic Interactive Mission Map UI**
✨ **Collision Detection & Autonomous Recovery**
✨ **Multiple Camera Views (Front / Rear / Bird Eye / Cinematic)**
✨ **Synchronous Deterministic Simulation**

---

# 🎥 System Architecture

```
              RGB Camera
                   │
                   ▼
          NVIDIA CNN Model
             (Steering)
                   │
                   ▼
Global Route Planner ───► Path Steering
                   │
                   ▼
           Sensor Fusion Layer
     (LiDAR + Traffic Light + Semantic)
                   │
                   ▼
             PID Controller
        (Throttle / Brake / Steer)
                   │
                   ▼
               CARLA Vehicle
```

---

# 🧠 Technologies Used

| Component       | Technology                               |
| --------------- | ---------------------------------------- |
| Simulator       | CARLA 0.9.10                             |
| Language        | Python                                   |
| Deep Learning   | PyTorch                                  |
| Computer Vision | OpenCV                                   |
| Path Planning   | CARLA GlobalRoutePlanner                 |
| Sensors         | RGB Camera, LiDAR, Semantic Segmentation |
| Control         | PID Controller                           |

---

# 📁 Project Structure

```
CARLA-Hybrid-Autonomous-System/

├── main.py
│   └── Complete autonomous driving pipeline
│
├── models/
│   └── converted_model.pth
│
├── README.md
│
└── assets/
    └── system_architecture.png
```

---

# ⚙️ Requirements

## Software

* Windows 10 / 11
* Python **3.7 – 3.9**
* CARLA **0.9.10**
* PyTorch
* OpenCV
* NumPy

---

# 🔧 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/CARLA-Hybrid-Autonomous-System.git
cd CARLA-Hybrid-Autonomous-System
```

---

## 2️⃣ Install Dependencies

```bash
pip install torch torchvision numpy opencv-python
```

---

## 3️⃣ Start CARLA Simulator

```
CARLA_0.9.10/WindowsNoEditor/CarlaUE4.exe
```

---

## 4️⃣ Update Paths

Inside **main.py**

```python
CARLA_ROOT = r"E:\CAR-simulator\CARLA_0.9.10\WindowsNoEditor"
MODEL_PATH = r"E:\CAR-simulator\models\kaggle\converted_model.pth"
```

---

# 🚀 Running the Autonomous System

```bash
python main.py
```

After launching you will see:

🪟 **Autonomous driving camera window**
🗺 **Interactive mission control map**
🚗 **Vehicle ready for mission**

---

# 🖱 Mission Control (Map UI)

| Action          | Control           |
| --------------- | ----------------- |
| Zoom            | Mouse Scroll      |
| Pan             | Middle Mouse Drag |
| Set Start       | Left Click        |
| Set Destination | Right Click       |
| Reset View      | R                 |
| Quit            | Q                 |

---

# 🎥 Camera Views

Press **V** to switch camera modes.

| View     | Description              |
| -------- | ------------------------ |
| Front    | Default driving camera   |
| Rear     | Reverse recovery camera  |
| Left     | Side view                |
| Right    | Side view                |
| Bird Eye | Top-down view            |
| Chase    | Cinematic following view |

---

# 🚦 Autonomous Driving Logic

The system uses **hybrid intelligence** combining:

## 🛣 Path Planning

Uses CARLA's **GlobalRoutePlanner**

Features:

* Waypoint-based routing
* Dynamic waypoint trimming
* Smooth trajectory tracking

---

## 🤖 Deep Learning Steering

Based on **NVIDIA Behavioral Cloning Model**

Input:

```
RGB Camera Image
```

Output:

```
Steering Angle
```

The CNN learns human driving behavior from training data.

---

## 📡 LiDAR Obstacle Detection

LiDAR scans the vehicle's front region and detects:

* Vehicles
* Pedestrians
* Crossing obstacles

Safety behaviors include:

* Emergency stop
* Speed reduction
* Obstacle avoidance steering

---

## 🚦 Traffic Light Handling

Vehicle detects traffic lights using CARLA API.

Behavior:

* Stop on **RED**
* Slow on **YELLOW**
* Continue on **GREEN**

---

## 🔄 Crash Recovery System

If collision occurs:

```
Collision Detected
        ↓
Reverse using rear camera
        ↓
Check road via semantic segmentation
        ↓
Realign with lane
        ↓
Resume mission
```

This avoids simulation resets and mimics **real autonomous recovery behavior**.

---

# 🗺 Dynamic Map Engine

The system includes a **fully interactive map UI**.

Features:

* Zoom & pan
* Live vehicle tracking
* Route visualization
* Start & End markers

---

# 🚗 Driving State Machine

```
IDLE
 ↓
CALC_TO_START
 ↓
TO_START
 ↓
CALC_TO_END
 ↓
TO_END
 ↓
FINISHED
```

Extra states:

```
RECOVERING
REALIGN
```

---

# 🧪 Tested Configuration

| Parameter       | Value         |
| --------------- | ------------- |
| Town            | Town03        |
| Vehicle         | Tesla Model 3 |
| FPS             | 20            |
| Simulation Mode | Synchronous   |

---

# ⚠️ Limitations

This project is designed for **research and learning purposes**.

Current limitations:

* No object detection model (YOLO)
* No pedestrian behavior prediction
* No multi-lane behavior planning
* No reinforcement learning

---

# 🚀 Future Improvements

Possible upgrades:

* YOLO object detection
* Transformer driving models
* Multi-lane planning
* RL-based driving policy
* Sensor fusion with radar
* Bird-Eye-View perception

---

# 👨‍💻 Author

**Nikshith**

🚗 Autonomous Driving Enthusiast
🧠 AI + Robotics Systems
💻 Python | PyTorch | CARLA

---

# ⭐ Final Thoughts

This project demonstrates how **modern autonomous driving systems combine**:

```
Deep Learning
+ Classical Planning
+ Sensor Fusion
```

to achieve **robust autonomous navigation in simulation**.

---

⭐ If you find this project useful, consider **starring the repository**!

---

