#version 150
in vec3 v_Pos;
in vec3 v_Normal;
in vec2 v_Uv;
out vec4 Target;
uniform vec3 view_pos;
uniform vec3 light_pos;
uniform float ambient_strength;
uniform sampler2D t_Texture;
uniform int spec_samples;

void main() {
  vec3 ambient = ambient_strength * vec3(1.0, 1.0, 1.0);
  vec3 norm = normalize(v_Normal);
  vec3 light_dir = normalize(light_pos - v_Pos);
  float diff = max(dot(norm, light_dir), 0.0);
  vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
  float specular_strength = 4.0;
  vec3 view_dir = normalize(view_pos - v_Pos);
  vec3 reflect_dir = reflect(-light_dir, norm);
  float spec = pow(max(dot(view_dir, reflect_dir), 0.0), spec_samples);
  vec3 specular = specular_strength * spec * vec3(1.0, 1.0, 1.0);
  vec4 conv = vec4(ambient + specular, 1.0);
  Target = conv * texture(t_Texture, v_Uv);
}
