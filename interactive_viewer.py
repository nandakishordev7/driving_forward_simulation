"""
interactive_viewer.py

Real-time interactive 3D viewer for the Gaussian scene.
Controls exactly like a game:

  Mouse drag (left)   -> rotate scene
  W / S               -> move forward / backward
  A / D               -> strafe left / right
  Q / E               -> move up / down
  Scroll wheel        -> zoom in / out
  R                   -> reset to Bird's Eye View
  ESC                 -> quit

Requirements:
  pip install pygame PyOpenGL PyOpenGL_accelerate
"""

import numpy as np
import sys
import os

try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print("[viewer] Missing dependencies. Install with:")
    print("  pip install pygame PyOpenGL PyOpenGL_accelerate")
    sys.exit(1)


# ── Camera state ──────────────────────────────────────────────────────────────

class FlyCamera:
    def __init__(self):
        self.reset()

    def reset(self):
        """Bird's Eye View default."""
        self.pos   = np.array([0.0, 0.0, 40.0])   # start high up
        self.yaw   = 0.0      # rotation around Z (left/right)
        self.pitch = -89.0    # looking straight down
        self.speed = 0.3
        self.sensitivity = 0.25

    def get_forward(self):
        yaw_r   = np.radians(self.yaw)
        pitch_r = np.radians(self.pitch)
        return np.array([
            np.cos(pitch_r) * np.cos(yaw_r),
            np.cos(pitch_r) * np.sin(yaw_r),
            np.sin(pitch_r)
        ])

    def get_right(self):
        yaw_r = np.radians(self.yaw)
        return np.array([np.sin(yaw_r), -np.cos(yaw_r), 0.0])

    def get_up(self):
        return np.cross(self.get_right(), self.get_forward())

    def move(self, keys):
        fwd   = self.get_forward()
        right = self.get_right()
        up    = np.array([0, 0, 1.0])
        spd   = self.speed

        if keys[K_w] or keys[K_UP]:    self.pos += fwd   * spd
        if keys[K_s] or keys[K_DOWN]:  self.pos -= fwd   * spd
        if keys[K_a] or keys[K_LEFT]:  self.pos -= right * spd
        if keys[K_d] or keys[K_RIGHT]: self.pos += right * spd
        if keys[K_q]:                   self.pos -= up    * spd
        if keys[K_e]:                   self.pos += up    * spd

    def rotate(self, dx, dy):
        self.yaw   += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        self.pitch  = np.clip(self.pitch, -89.0, 89.0)

    def apply(self):
        """Set OpenGL modelview matrix from camera state."""
        target = self.pos + self.get_forward()
        up     = np.array([0, 0, 1.0])
        gluLookAt(
            self.pos[0],   self.pos[1],   self.pos[2],
            target[0],     target[1],     target[2],
            up[0],         up[1],         up[2],
        )


# ── Point cloud loader ────────────────────────────────────────────────────────

def prepare_pointcloud(gaussians, z_min=-2.5, z_max=3.0, bev_range=50.0,
                        max_points=500_000):
    """
    Filter and prepare Gaussian centres + colors for OpenGL rendering.
    Returns (vertices, colors) as contiguous float32 arrays.
    """
    xyz  = gaussians['xyz'] if isinstance(gaussians['xyz'], np.ndarray) \
           else gaussians['xyz'].cpu().numpy()
    sh   = gaussians['sh_coeffs'] if isinstance(gaussians['sh_coeffs'], np.ndarray) \
           else gaussians['sh_coeffs'].cpu().numpy()

    rgb  = np.clip(0.5 + 0.282 * sh[:, :3], 0, 1).astype(np.float32)

    # Z and radial clamp
    mask = ((xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max) &
            (np.abs(xyz[:, 0]) <= bev_range) &
            (np.abs(xyz[:, 1]) <= bev_range))
    xyz, rgb = xyz[mask], rgb[mask]

    # Subsample if too many points for real-time
    if len(xyz) > max_points:
        idx  = np.random.choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    print(f"[viewer] Loaded {len(xyz):,} points for rendering")
    return xyz.astype(np.float32), rgb.astype(np.float32)


# ── OpenGL helpers ────────────────────────────────────────────────────────────

def draw_grid(size=50, step=5):
    """Draw a reference grid on the XY plane."""
    glLineWidth(0.5)
    glColor4f(0.2, 0.2, 0.2, 1.0)
    glBegin(GL_LINES)
    for i in range(-size, size + 1, step):
        glVertex3f(i,    -size, 0)
        glVertex3f(i,     size, 0)
        glVertex3f(-size, i,    0)
        glVertex3f( size, i,    0)
    glEnd()


def draw_ego_vehicle():
    """Draw a green box at the origin representing the ego vehicle."""
    glLineWidth(2.0)
    glColor3f(0.0, 1.0, 0.0)

    # Car box: 4.5m long, 2m wide, 1.5m tall
    l, w, h = 2.25, 1.0, 1.5
    verts = [
        (-w, -l, 0), ( w, -l, 0), ( w,  l, 0), (-w,  l, 0),  # bottom
        (-w, -l, h), ( w, -l, h), ( w,  l, h), (-w,  l, h),  # top
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7),  # sides
    ]
    glBegin(GL_LINES)
    for e in edges:
        for v in e:
            glVertex3fv(verts[v])
    glEnd()

    # Forward arrow
    glLineWidth(3.0)
    glColor3f(0.0, 0.9, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0, l, h / 2)
    glVertex3f(0, l + 3, h / 2)
    glEnd()


def draw_range_rings():
    """Draw flat circles at 10m, 20m, 30m, 40m."""
    glLineWidth(0.5)
    glColor4f(0.25, 0.25, 0.25, 1.0)
    for r in [10, 20, 30, 40]:
        glBegin(GL_LINE_LOOP)
        for i in range(64):
            a = 2 * np.pi * i / 64
            glVertex3f(r * np.cos(a), r * np.sin(a), 0)
        glEnd()


def draw_axis():
    """Small XYZ axis indicator at origin."""
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glColor3f(1, 0, 0); glVertex3f(0,0,0); glVertex3f(3,0,0)   # X red
    glColor3f(0, 1, 0); glVertex3f(0,0,0); glVertex3f(0,3,0)   # Y green
    glColor3f(0, 0, 1); glVertex3f(0,0,0); glVertex3f(0,0,3)   # Z blue
    glEnd()


# ── Main viewer loop ──────────────────────────────────────────────────────────

def launch_viewer(gaussians, window_w=1280, window_h=720,
                  point_size=6.0, bg_color=(0.05, 0.05, 0.08)):
    """
    Launch the interactive 3D viewer.

    Args:
        gaussians  : merged Gaussian dict (ego frame, numpy or torch)
        window_w/h : window resolution
        point_size : GL point size
        bg_color   : background RGB tuple [0,1]
    """
    vertices, colors = prepare_pointcloud(gaussians)

    pygame.init()
    pygame.display.set_caption(
        "DrivingForward 3D Viewer  |  WASD=move  Mouse=rotate  R=reset  ESC=quit"
    )
    screen = pygame.display.set_mode(
        (window_w, window_h),
        DOUBLEBUF | OPENGL
    )
    pygame.event.set_grab(False)

    # OpenGL setup
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPointSize(point_size)
    glClearColor(*bg_color, 1.0)

    # Projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, window_w / window_h, 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)

    # Upload point cloud to GPU via VBO for fast rendering
    try:
        from OpenGL.arrays import vbo
        pos_vbo = vbo.VBO(vertices)
        col_vbo = vbo.VBO(colors)
        USE_VBO = True
        print("[viewer] Using VBO (fast path)")
    except Exception:
        USE_VBO = False
        print("[viewer] VBO unavailable, using immediate mode")

    camera     = FlyCamera()
    mouse_down = False
    last_mouse = (0, 0)
    clock      = pygame.time.Clock()

    print("\n[viewer] Controls:")
    print("  Mouse drag       -> rotate")
    print("  W/S              -> forward / back")
    print("  A/D              -> strafe left / right")
    print("  Q/E              -> up / down")
    print("  Scroll           -> zoom (speed adjust)")
    print("  R                -> reset to Bird's Eye View")
    print("  ESC              -> quit\n")

    running = True
    while running:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_r:
                    camera.reset()
                    print("[viewer] Camera reset to BEV")
                elif event.key == K_EQUALS or event.key == K_PLUS:
                    point_size = min(point_size + 0.5, 20.0)
                    glPointSize(point_size)
                    print(f"[viewer] Point size: {point_size:.1f}")
                elif event.key == K_MINUS:
                    point_size = max(point_size - 0.5, 0.5)
                    glPointSize(point_size)
                    print(f"[viewer] Point size: {point_size:.1f}")

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:   # left click
                    mouse_down = True
                    last_mouse = event.pos
                elif event.button == 4:  # scroll up = speed up
                    camera.speed = min(camera.speed * 1.2, 5.0)
                elif event.button == 5:  # scroll down = slow down
                    camera.speed = max(camera.speed / 1.2, 0.05)

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False

            elif event.type == MOUSEMOTION:
                if mouse_down:
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]
                    camera.rotate(dx, dy)
                    last_mouse = event.pos

        # Keyboard movement
        keys = pygame.key.get_pressed()
        camera.move(keys)

        # Render frame
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        camera.apply()

        # Scene elements
        draw_grid()
        draw_range_rings()
        draw_ego_vehicle()
        draw_axis()

        # Point cloud
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        if USE_VBO:
            pos_vbo.bind()
            glVertexPointer(3, GL_FLOAT, 0, pos_vbo)
            col_vbo.bind()
            glColorPointer(3, GL_FLOAT, 0, col_vbo)
        else:
            glVertexPointer(3, GL_FLOAT, 0, vertices)
            glColorPointer(3, GL_FLOAT, 0, colors)

        glDrawArrays(GL_POINTS, 0, len(vertices))

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        if USE_VBO:
            pos_vbo.unbind()
            col_vbo.unbind()

        # HUD
        fps = clock.get_fps()
        pygame.display.set_caption(
            f"DrivingForward 3D  |  {fps:.0f} FPS  |  "
            f"pos=({camera.pos[0]:.1f}, {camera.pos[1]:.1f}, {camera.pos[2]:.1f})  |  "
            f"pitch={camera.pitch:.0f}  yaw={camera.yaw:.0f}  |  "
            f"pts={point_size:.1f}px (+/-)  speed={camera.speed:.2f}  R=reset  ESC=quit"
        )

        pygame.display.flip()

    pygame.quit()
    print("[viewer] Closed.")