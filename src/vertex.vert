#version 150
in vec3 a_Pos;
in vec3 a_Normal;
in vec2 a_Uv;
out vec3 v_Pos;
out vec3 v_Normal;
out vec2 v_Uv;
uniform mat4 u_View;
uniform mat4 u_Proj;
uniform sampler2D t_Texture;

void main() {
  v_Pos = a_Pos;
  v_Normal = a_Normal;
  v_Uv = a_Uv;
  gl_Position = u_Proj * u_View * vec4(a_Pos, 1.0);
}
