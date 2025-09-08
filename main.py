import numpy as np
import fdtd
fdtd.set_backend("numpy.float16")

import arcade
from fdtd import Grid

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
WINDOW_TITLE = "FDTD GPU Shader Demo"

GRID_SCALE = 2
WAVELENGTH = 1
SPEED_LIGHT = 299792458.0


class GameView(arcade.View):
    def __init__(self):
        super().__init__()
        self.background_color = arcade.color.BLACK

        self.grid = Grid(
            shape=(WINDOW_WIDTH // GRID_SCALE, WINDOW_HEIGHT // GRID_SCALE, 1),
            grid_spacing=WAVELENGTH,
            permittivity=1.0,
            permeability=1.0,
        )

        self.grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
        self.grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")
        self.grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
        self.grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")
        self.grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")

        self.id_N = 0

        self.ctx = self.window.ctx
        self.quad = arcade.gl.geometry.quad_2d_fs()

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert;
                out vec2 uv;
                void main() {
                    uv = in_vert * 0.5 + 0.5;   // map from [-1,1] → [0,1]
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D field_tex;
                uniform int blur_radius;
                uniform vec3 background_color = vec3(0,0,0);   // NEW: background color
                in vec2 uv;
                out vec4 fragColor;
                
                void main() {
                    vec2 texSize = vec2(textureSize(field_tex, 0));
                    vec2 offset = 1.0 / texSize;
                
                    float sum = 0.0;
                    int count = 0;
                
                    // Blur pass
                    for (int x = -blur_radius; x <= blur_radius; x++) {
                        for (int y = -blur_radius; y <= blur_radius; y++) {
                            vec2 uv_offset = uv + vec2(x, y) * offset;
                            sum += texture(field_tex, uv_offset).r;
                            count++;
                        }
                    }
                
                    float val = sum / float(count);
                
                    // Clamp [-1,1]
                    val = clamp(val, -1.0, 1.0);
                
                    // Color mapping
                    vec3 baseColor;
                    if (val >= 0.0) {
                        baseColor = mix(vec3(0,0,0), vec3(1), val); // blue → white
                    } else {
                        baseColor = mix(vec3(1,0,0), vec3(1), -val); // red → white
                    }
                
                    // Bloom glow
                    float brightness = max(baseColor.r, max(baseColor.g, baseColor.b));
                    vec3 glow = baseColor * smoothstep(0.6, 1.0, brightness);
                    glow *= 1.5;
                
                    // Combine with background
                    vec3 color = mix(background_color, baseColor + glow, 1.0);
                
                    fragColor = vec4(color, 1.0);
                }
                """
        )

        self.field_tex = self.ctx.texture(
            (self.grid.Nx, self.grid.Ny),
            components=1,
            dtype="f4",
        )

        self.blur_radius = 1
        self.prog["blur_radius"] = self.blur_radius

    def on_draw(self):
        self.clear()
        self.field_tex.use(0)
        self.quad.render(self.prog)

    def on_update(self, delta_time):
        self.grid.step()
        Ez = self.grid.E[:, :, 0, 2]
        maxval = np.max(np.abs(Ez)) + 1e-9
        field = (Ez / maxval).astype("f4")
        self.field_tex.write(field.tobytes())

    def on_mouse_press(self, x, y, button, modifiers):
        gx = int(x // GRID_SCALE)
        gy = -int((WINDOW_HEIGHT - y) // GRID_SCALE)  # flip Y
        self.grid[gy, gx, 0] = fdtd.PointSource(
            period=WAVELENGTH / SPEED_LIGHT,
            name=f"pointsource{self.id_N}",
        )
        self.id_N += 1

    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.UP:
            self.blur_radius = min(self.blur_radius + 1, 10)  # limit max radius
            self.prog["blur_radius"] = self.blur_radius

        elif symbol == arcade.key.DOWN:
            self.blur_radius = max(self.blur_radius - 1, 0)
            self.prog["blur_radius"] = self.blur_radius


def main():
    window = arcade.Window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    game = GameView()
    window.show_view(game)
    arcade.run()


if __name__ == "__main__":
    main()
