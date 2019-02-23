#![windows_subsystem = "windows"]
extern crate cgmath;
extern crate gl;
extern crate glutin;
extern crate image;
extern crate rand;
extern crate tobj;
use glutin::*;
use cgmath::*;
use gl::types::*;
use std::thread;
use std::time::Duration;
use std::sync::mpsc::*;
use std::thread::sleep;
use std::os::raw::c_void;
use std::ffi::CString;
use std::ptr;
use std::ops::Deref;
use std::path::Path;
use std::mem::size_of;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
///Default user input camera movement speed ratio
const CAM_SPEED: f32 = 4.;
///Defaults autocamera off
const AUTOCAMERA: bool = false;
///Wait time for each thread, to stop them murdering the CPU
const WAIT_TIME: u64 = 10;

///Typedef for a heap allocated pointer to a function that works on the OpenGL owning thread
type GLFunc = Box<Fn(&mut GLDevice, &mut Window) + Send>;
///Typedef for a heap allocated pointer to a function that works on the Camera thread
type CamFunc = Box<Fn(&mut Camera) + Send>;
///Typedef for a heap allocated pointer to a function that works on the Game (main) thread
type GameFunc = Box<Fn(&mut Game) + Send>;

///Links the OpenGL shaders into programmes
fn gen_shader_programme<T: Into<Vec<u8>>>(vertex: T, fragment: T) -> u32 {
    unsafe {
        let shader_programme = gl::CreateProgram();
        let vertex = gen_shader(vertex, ShaderType::Vertex);
        let fragment = gen_shader(fragment, ShaderType::Fragment);
        gl::AttachShader(shader_programme, vertex);
        gl::AttachShader(shader_programme, fragment);
        gl::LinkProgram(shader_programme);
        gl::DetachShader(shader_programme, vertex);
        gl::DetachShader(shader_programme, fragment);
        gl::DeleteShader(vertex);
        gl::DeleteShader(fragment);
        shader_programme
    }
}

enum ShaderType {
    Vertex,
    Fragment,
}
///Compiles the shader
fn gen_shader<T: Into<Vec<u8>>>(code: T, shader_type: ShaderType) -> u32 {
    let c_str = CString::new(code).unwrap();
    unsafe {
        let shader = {
            match shader_type {
                ShaderType::Vertex => gl::CreateShader(gl::VERTEX_SHADER),
                ShaderType::Fragment => gl::CreateShader(gl::FRAGMENT_SHADER),
            }
        };
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), ptr::null());
        gl::CompileShader(shader);
        shader
    }
}
///Primarily controls the window title, can be autodereferenced to become a glutin::GlWindow
struct Window {
    update: bool,
    level: usize,
    moves: usize,
    window: glutin::GlWindow,
}
impl Window {
    ///Set window title based on levels and moves from the window struct, called when Window's update is true
    fn update_window(&self) {
        self.window
            .set_title(&format!("Level: {} Moves: {}", self.level, self.moves));
    }
}

///Implements deref trait for the Window structure, which allows a pointer to my Window struct to be implicitly
///converted to a pointer to the inner glutin::GlWindow
impl Deref for Window {
    type Target = glutin::GlWindow;
    fn deref(&self) -> &Self::Target {
        &self.window
    }
}

///Structure to store various buffer id's for talking to the GPU.
struct GLDevice {
    vao: u32,
    vbo: u32,
    ebo: u32,
    texture: [u32; 5],
    shader_programme: u32,
}
///Destructor for GLDevice to clean up GPU
impl Drop for GLDevice {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.shader_programme);
            gl::DeleteTextures(5, self.texture[0] as *const _);
            gl::DeleteBuffers(1, self.vbo as *const _);
            gl::DeleteBuffers(1, self.ebo as *const _);
            gl::DeleteVertexArrays(1, self.vao as *const _);
        }
    }
}
///Trait for converting data to the correct input type, generally void*'s
trait AsDataPointer {
    type Pointer;
    fn as_data_ptr(&self) -> *const Self::Pointer;
}

///Structure for each vertex
///Implements Debug, which could be removed now but useful for debugging while producing the code
///it also implements clone for duplicating the data and copy for specifying this type can just be
///simply memcopyed rather than requiring a deep copy
///
#[derive(Debug, Clone, Copy)]
struct Vertex {
    ///position of the vertex in 3d space
    pos: [f32; 3],
    ///normal coordinates for lighting calculations
    normal: [f32; 3],
    ///texture coordinates
    uv: [f32; 2],
}
///Trait for converting from the format the .obj model loader tobj gives me to my own internal formats
trait FromOBJ {
    fn from_obj_model(&self) -> (Vec<Vertex>, Vec<u32>);
}
impl FromOBJ for tobj::Model {
    ///Takes reference to a tobj::Model, converts it to my Vertex structure + index offsets, ignoring
    ///unneccesary data
    fn from_obj_model(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut vec = Vec::new();
        for i in 0..self.mesh.positions.len() / 3 {
            vec.push(Vertex {
                pos: [
                    self.mesh.positions[i * 3],
                    self.mesh.positions[i * 3 + 1],
                    self.mesh.positions[i * 3 + 2],
                ],
                uv: [self.mesh.texcoords[i * 2], self.mesh.texcoords[i * 2 + 1]],
                normal: [
                    self.mesh.normals[i * 3],
                    self.mesh.normals[i * 3 + 1],
                    self.mesh.normals[i * 3 + 2],
                ],
                ..Default::default()
            });
        }
        (vec, self.mesh.indices.clone())
    }
}

impl AsDataPointer for Vertex {
    type Pointer = c_void;
    fn as_data_ptr(&self) -> *const Self::Pointer {
        &self.pos[0] as *const f32 as *const c_void
    }
}
impl AsDataPointer for u32 {
    type Pointer = c_void;
    fn as_data_ptr(&self) -> *const Self::Pointer {
        self as *const u32 as *const _
    }
}
impl AsDataPointer for f32 {
    type Pointer = c_void;
    fn as_data_ptr(&self) -> *const Self::Pointer {
        self as *const f32 as *const _
    }
}
impl AsDataPointer for cgmath::Matrix4<f32> {
    type Pointer = f32;
    fn as_data_ptr(&self) -> *const Self::Pointer {
        &self.x.x as *const f32
    }
}
///Implements AsDataPointer for any vector of items containing AsDataPointer
///by calling the first element in the vectors as_data_ptr() method
impl<T: AsDataPointer> AsDataPointer for Vec<T> {
    ///Specifies to use the Pointer type of the contained type
    type Pointer = T::Pointer;
    fn as_data_ptr(&self) -> *const Self::Pointer {
        self[0].as_data_ptr()
    }
}
///Implements a default constructor that can be overloaded with specified values
impl Default for Vertex {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            normal: [1.0; 3],
            uv: [0.0; 2],
        }
    }
}
///Camera structure for generating view matrix's and sending them to the OpenGL thread
struct Camera {
    ///Position of the camera in 3d space
    camera_pos: cgmath::Vector3<f32>,
    ///Directon the camera is pointing in
    camera_front: cgmath::Vector3<f32>,
    ///Provides a constant axis to work around
    camera_up: cgmath::Vector3<f32>,
    ///Receives commands sent to the camera
    receiver: Receiver<CamFunc>,
    ///Sends commands to the OpenGL thread
    gl_sender: Sender<GLFunc>,
    ///A timer since the the structure was created
    start_time: std::time::Instant,
    ///If the camera is in autocamera mode, how fast does it move
    auto_cam_speed: f32,
    ///If the camera moves of its own accord rather than only user input
    autocamera: bool,
}

impl Camera {
    ///Generates the camera structure
    fn new(receiver: Receiver<CamFunc>, sender: Sender<GLFunc>) -> Self {
        let camera_pos = cgmath::vec3(1f32, 7.0, -10.0);
        let camera_front = cgmath::vec3(1.0, 0.0, -1.0);
        let camera_up = cgmath::vec3(0.0, 1.0, 0.0);
        Self {
            camera_front,
            camera_pos,
            camera_up,
            gl_sender: sender,
            receiver,
            autocamera: AUTOCAMERA,
            auto_cam_speed: 1.0,
            start_time: std::time::Instant::now(),
        }
    }
    ///If the autocamera bool is true, this is called once per 10ms to modify
    ///the camera position
    fn autocamera(&mut self) {
        let time = {
            let now = |t: &std::time::Instant| {
                t.elapsed().as_secs() as f32
                    + self.start_time.elapsed().subsec_nanos() as f32 * 1e-9
            };
            now(&self.start_time)
        };
        self.camera_pos.x = 15.0_f32 * (time * self.auto_cam_speed).sin();
        self.camera_pos.z = 15.0_f32 * (time * self.auto_cam_speed).cos();
    }
    ///Generates a view matrix which is sent to the shader and used to move
    ///the vertices by the GPU
    fn view(&self) -> Matrix4<f32> {
        let view = cgmath::Matrix4::look_at(
            Point3::from_vec(self.camera_pos),
            Point3::from_vec(self.camera_front),
            self.camera_up,
        );
        view
    }
    ///Generates view matrix and sends it to the GPU thread
    fn send_view(&mut self) {
        {
            let view_matrix = self.view();
            self.gl_sender
                .send(Box::new({
                    let pos = self.camera_pos.clone();
                    move |device, _| unsafe {
                        gl::UniformMatrix4fv(
                            gl::GetUniformLocation(
                                device.shader_programme,
                                CString::new("u_View").unwrap().as_ptr(),
                            ),
                            1,
                            gl::FALSE,
                            view_matrix.as_data_ptr(),
                        );

                        gl::Uniform3f(
                            gl::GetUniformLocation(
                                device.shader_programme,
                                CString::new("view_pos").unwrap().as_ptr(),
                            ),
                            pos[0],
                            pos[1],
                            pos[2],
                        );
                    }
                }))
                .unwrap();
        }
    }
    ///Thread locking loop where the camera polls its own receiver for inputs
    ///and deals with them
    fn run(&mut self) {
        loop {
            sleep(Duration::from_millis(WAIT_TIME));
            let mut needs_update = {
                if self.autocamera {
                    self.autocamera();
                    true
                } else {
                    false
                }
            };
            while let Ok(func) = self.receiver.try_recv() {
                needs_update = true;
                func(self)
            }
            if needs_update {
                self.send_view()
            }
        }
    }
}
impl GLMove for Camera {
    fn translate(&mut self, x: f32, y: f32, z: f32) {
        let (x, y, z) = (x * CAM_SPEED, y * CAM_SPEED, z * CAM_SPEED);
        self.camera_pos.x += x;
        self.camera_front.x += x;
        self.camera_pos.y += y;
        self.camera_front.y += y;
        self.camera_pos.z += z;
        self.camera_front.z += z;
    }
}
///Main Game structure that holds most data to do with the actual game logic
struct Game {
    ///Should the game close
    running: bool,
    ///Storage for the default object meshes, i use this to refresh their position
    ///level reload rather than having to track the translations and reverse them
    meshes: Vec<Object>,
    ///Player object
    player: Player,
    ///Walls
    walls: Vec<Wall>,
    ///Crates
    crates: Vec<Crate>,
    ///The markers to show where the crates should go
    floor_tiles: Vec<FloorMarker>,
    ///Sends commands to the OpenGL owning thread
    gl_sender: Sender<GLFunc>,
    ///Sends commands to the Camera thread
    camera_sender: Sender<CamFunc>,
    ///Receives commands for the Game thread (main thread)
    game_receiver: Receiver<GameFunc>,
    ///Current level
    current_level: isize,
    ///Stores levels
    ///This is a vector of functions that alter this Game structure
    ///This allows the level order and level count to be dynamic if wanted
    levels: Vec<fn(&mut Game)>,
    ///This stores a hashmap for bools.
    ///This allows level functions to create threads with names (the hashmap key),
    ///which allows dynamic change throughout the level.
    ///These keys can then be checked by the spawned thread to check if they should
    ///terminate\n
    ///These are atomically reference counted and encapsulated in an access lock for
    /// thread safety
    subroutine: HashMap<isize, Arc<RwLock<bool>>>,
}

enum Direction {
    Up,
    Down,
    Left,
    Right,
}
///Loads level zero
fn level_zero(g: &mut Game) {
    g.typical_reset();
    g.uniform_reset();
    g.drop_key(3).unwrap_or_else(|error| println!("{}", error));
    g.drop_key(4).unwrap_or_else(|error| println!("{}", error));
    g.crates[0].move_obj(-1, 1);
    g.crates[1].move_obj(1, 1);
    g.floor_tiles[0].move_obj(0, 1);
    g.floor_tiles[1].move_obj(2, 1);
    g.current_level = 0;
}
///Loads level four
fn level_four(g: &mut Game) {
    g.typical_reset();
    g.uniform_reset();
    g.player.move_obj(-3, -3);
    g.crates[0].move_obj(-1, 1);
    g.crates[1].move_obj(1, 1);
    g.floor_tiles[0].move_obj(0, -2);
    g.floor_tiles[0].move_obj(1, 2);
    let walls = vec![[-1, -1], [2, 2], [-2, -2], [0, 3], [1, 3], [3, 3]];
    g.load_walls(&walls[..]);
    let proceed = g.subroutine.get(&4).unwrap().clone();
    *proceed.write().expect("error in mutex @ level 4") = true;
    let camera_sender = g.camera_sender.clone();
    thread::spawn(move || {
        use rand::distributions::{IndependentSample, Range};
        let between = Range::new(-7., 7.);
        let mut rng = rand::thread_rng();
        while *proceed.read().unwrap() {
            thread::sleep(Duration::from_secs(2));
            let cam_speed = between.ind_sample(&mut rng);
            camera_sender
                .send(Box::new(move |camera: &mut Camera| {
                    camera.autocamera = true;
                    camera.auto_cam_speed = cam_speed;
                }))
                .unwrap();
        }
        camera_sender
            .send(Box::new(move |camera: &mut Camera| {
                camera.autocamera = false;
                camera.auto_cam_speed = 1.;
            }))
            .unwrap();
    });
}
///Loads level three
fn level_three(g: &mut Game) {
    {
        g.typical_reset();
        g.player.move_obj(1, 0);
        g.crates[0].move_obj(-2, 1);
        g.crates[1].move_obj(0, 1);
        g.floor_tiles[0].move_obj(0, -2);
        g.floor_tiles[0].move_obj(3, -1);
        let walls = vec![[2, 1], [1, 1], [1, 2], [3, 3], [3, 2], [3, 1]];
        g.load_walls(&walls);
        let gl_sender = g.gl_sender.clone();
        let proceed = Arc::clone(g.subroutine.get(&3).unwrap());
        *proceed.write().expect("error level 3 @ mutex") = true;
        thread::spawn(move || {
            let mut light_position = [0.0, 10.0, 10.0];
            use rand::distributions::{IndependentSample, Range};
            let between = Range::new(-0.05, 0.05);
            let mut rng = rand::thread_rng();
            let (mut counter, mut offset) = (0, [0.; 3]);
            gl_sender
                .send(Box::new(move |device, _| unsafe {
                    gl::Uniform1f(
                        gl::GetUniformLocation(
                            device.shader_programme,
                            CString::new("ambient_strength").unwrap().as_ptr(),
                        ),
                        0.1,
                    );
                    gl::Uniform1i(
                        gl::GetUniformLocation(
                            device.shader_programme,
                            CString::new("spec_samples").unwrap().as_ptr(),
                        ),
                        2,
                    );
                    gl::Uniform3f(
                        gl::GetUniformLocation(
                            device.shader_programme,
                            CString::new("light_pos").unwrap().as_ptr(),
                        ),
                        0.0,
                        10.0,
                        10.0,
                    )
                }))
                .unwrap();
            thread::sleep(Duration::from_millis(WAIT_TIME));
            while *proceed.read().unwrap() {
                thread::sleep(Duration::from_millis(WAIT_TIME));
                if counter == 0 {
                    counter = 120;
                    offset[0] = between.ind_sample(&mut rng);
                    offset[1] = between.ind_sample(&mut rng);
                    offset[2] = between.ind_sample(&mut rng);
                }
                counter -= 1;
                if light_position[0] > 7.0 {
                    offset[0] -= 0.1;
                }
                if light_position[0] < -7.0 {
                    offset[0] += 0.1;
                }
                if light_position[1] > 7.0 {
                    offset[1] -= 0.1;
                }
                if light_position[1] < 2.0 {
                    offset[1] += 0.1;
                }
                if light_position[2] > 7.0 {
                    offset[2] -= 0.1;
                }
                if light_position[2] < -7.0 {
                    offset[2] += 0.1;
                }
                light_position[0] += offset[0];
                light_position[1] += offset[1];
                light_position[2] += offset[2];
                gl_sender
                    .send(Box::new(move |device, _| unsafe {
                        gl::Uniform3f(
                            gl::GetUniformLocation(
                                device.shader_programme,
                                CString::new("light_pos").unwrap().as_ptr(),
                            ),
                            light_position[0],
                            light_position[1],
                            light_position[2],
                        )
                    }))
                    .unwrap();
            }
        });
    }
}
///Loads level one
fn level_one(g: &mut Game) {
    g.camera_sender
        .send(Box::new(move |cam| {
            cam.autocamera = true;
        }))
        .unwrap();
    g.typical_reset();
    g.player.move_obj(0, 0);
    g.crates[0].move_obj(1, -1);
    g.crates[1].move_obj(0, -1);
    let walls = vec![[2, 2], [3, 2], [4, 2], [-3, -3], [-2, 2], [1, 1]];
    g.load_walls(&walls);
    g.floor_tiles[0].move_obj(0, 1);
}

///Loads level two
fn level_two(g: &mut Game) {
    g.gl_sender
        .send(Box::new(move |device, _| unsafe {
            gl::Uniform1f(
                gl::GetUniformLocation(
                    device.shader_programme,
                    CString::new("ambient_strength").unwrap().as_ptr(),
                ),
                0.1,
            );
        }))
        .unwrap();
    g.typical_reset();
    g.player.move_obj(0, -2);
    g.crates[0].move_obj(1, -1);
    g.crates[1].move_obj(0, -1);
    let walls = vec![[-2, 2], [-3, 2], [-1, 2], [-3, -3], [-2, 2], [1, 1]];
    g.load_walls(&walls);
    g.floor_tiles[0].move_obj(0, 1);
}

impl Game {
    ///Resets the adjustable walls
    fn load_walls(&mut self, walls: &[[i32; 2]]) {
        &self.walls
            .iter_mut()
            .take(10)
            .zip(walls.iter())
            .for_each(|(a, b)| {
                a.move_obj(4, 4);
                a.move_obj(b[0], b[1]);
            });
    }
    ///Some common reset actions for all levels
    fn typical_reset(&mut self) {
        self.player.reset(&self.meshes[1]);

        for mut i in &mut self.crates {
            i.reset(&self.meshes[2]);
        }
        for mut i in &mut self.floor_tiles {
            i.reset(&self.meshes[3]);
        }
        for i in 0..10 {
            self.walls[i].reset(&self.meshes[2]);
            self.walls[i].move_obj(-4, -4);
        }
    }
    ///Sends some default values to OpenGL
    fn uniform_reset(&self) {
        self.camera_sender
            .send(Box::new(move |camera| {
                camera.autocamera = false;

                camera.camera_pos = cgmath::vec3(1f32, 7.0, -10.0);
                camera.camera_front = cgmath::vec3(1.0, 0.0, -1.0);
                camera.camera_up = cgmath::vec3(0.0, 1.0, 0.0);
                camera.auto_cam_speed = 1.;
            }))
            .unwrap();
        self.gl_sender
            .send(Box::new(move |device, _| unsafe {
                gl::Uniform3f(
                    gl::GetUniformLocation(
                        device.shader_programme,
                        CString::new("light_pos").unwrap().as_ptr(),
                    ),
                    0.,
                    20.,
                    0.,
                );
                gl::Uniform1f(
                    gl::GetUniformLocation(
                        device.shader_programme,
                        CString::new("ambient_strength").unwrap().as_ptr(),
                    ),
                    0.7,
                );
                gl::Uniform1i(
                    gl::GetUniformLocation(
                        device.shader_programme,
                        CString::new("spec_samples").unwrap().as_ptr(),
                    ),
                    2,
                );
            }))
            .unwrap();
    }
    ///Kills a thread created by one of the level functions.
    ///This returns a Result monad which can either be an empty Ok value or a
    ///Err string reference type.
    ///This is then handled by the caller
    fn drop_key<'a>(&mut self, key: isize) -> Result<(), &'a str> {
        if self.subroutine.contains_key(&key) {
            *self.subroutine[&key].write().unwrap() = false;
            Ok(())
        } else {
            Err("Key not found")
        }
    }
    ///Creates a new game object
    pub fn new(
        meshes: Vec<Object>,
        player: Player,
        gl_sender: Sender<GLFunc>,
        camera_sender: Sender<CamFunc>,
        game_receiver: Receiver<GameFunc>,
    ) -> Self {
        Self {
            meshes,
            walls: Vec::new(),
            crates: Vec::new(),
            floor_tiles: Vec::new(),
            gl_sender,
            camera_sender,
            game_receiver,
            player,
            current_level: -1,
            levels: Vec::new(),
            subroutine: HashMap::new(),
            running: true,
        }
    }
    ///Locks this thread, polls inputs, handles messages sent to the
    ///main thread
    pub fn run(&mut self, events: &mut Events) {
        while self.running {
            events.poll();
            while let Ok(x) = self.game_receiver.try_recv() {
                x(self)
            }
            thread::sleep(Duration::from_millis(WAIT_TIME));
        }
    }
    ///Checks the status of the crates, if the game's won, unwinnable etc
    pub fn check(&mut self) {
        if self.crates
            .iter()
            .all(|x| (x.correct)(&x, &self.floor_tiles[..]))
        {
            self.current_level += 1;
            self.load_level();
        } else {
            let a = self.crates
                .iter()
                .map(|x| {
                    if !(x.correct)(&x, &self.floor_tiles) {
                        (x.screwed)(&x, &self.walls[..], &self.crates[..])
                    } else {
                        (true, false)
                    }
                })
                .collect::<Vec<_>>();

            if a.iter().any(|&(_, x)| x == true) || a.iter().all(|&(x, _)| x == true) {
                self.current_level = 0;
                self.load_level();
            }
        }
    }

    ///Loads a new level
    pub fn load_level(&mut self) {
        if self.current_level < self.levels.len() as isize {
            self.levels[self.current_level as usize](self);
            self.gl_sender
                .send(Box::new({
                    let current_level = self.current_level;
                    move |_, window: &mut Window| {
                        window.level = current_level as usize;
                        window.moves = 0;
                        window.update = true;
                    }
                }))
                .unwrap();
        } else {
            self.gl_sender
                .send(Box::new({
                    move |_, window: &mut Window| {
                        window.set_title("You win!");
                    }
                }))
                .unwrap();
            self.current_level = 0;
            thread::sleep(Duration::from_secs(5));
            self.load_level();
        }
    }
    ///Translates the player based on the direction enum sent from the events loop
    pub fn move_player(&mut self, direction: Direction) {
        let current_position = self.player.position();
        use Direction::*;
        let dir_values = match direction {
            Up => [0, 1],
            Down => [0, -1],
            Left => [1, 0],
            Right => [-1, 0],
        };

        let proposed_position = [
            current_position[0] + dir_values[0],
            current_position[1] + dir_values[1],
        ];

        if let Some(_) = collision_check(proposed_position.clone(), &self.walls[..]) {
            //Ok we've hit a wall we're done here
            return;
        }
        if let Some(collision_crate_index) =
            collision_check(proposed_position.clone(), &self.crates[..])
        {
            let proposed_crate_position = [
                proposed_position[0] + dir_values[0],
                proposed_position[1] + dir_values[1],
            ];

            if collision_check(proposed_crate_position.clone(), &self.walls[..]) == None
                && collision_check(proposed_crate_position.clone(), &self.crates[..]) == None
            {
                //Nothing behind box
                self.crates[collision_crate_index].move_obj(dir_values[0], dir_values[1]);
                self.player.move_obj(dir_values[0], dir_values[1]);
            }
        } else {
            //Move player
            self.player.move_obj(dir_values[0], dir_values[1]);
            //Update window title
            self.gl_sender
                .send(Box::new(move |_, w: &mut Window| {
                    w.update = true;
                    w.moves += 1;
                }))
                .unwrap();
        }
        self.check();
    }
}
///This checks to see if the proposed new position of an object is in collision with any other
///based on a slice of objects that implement the Entity trait.
///That is to say its statically polymorphic over anything "inheriting" from Entity
///It then either returns None or returns the index of the colliding object from the slice
fn collision_check<T: Entity>(proposed_position: [i32; 2], entity: &[T]) -> Option<usize> {
    for (index, i) in entity.iter().enumerate() {
        if i.is_active() {
            if i.position() == proposed_position {
                return Some(index);
            }
        }
    }
    None
}

impl Entity for Player {
    fn position(&self) -> [i32; 2] {
        self.position
    }
    fn gl(&self) -> Object {
        self.gl.clone()
    }
    fn gl_send(&self) -> &Sender<GLFunc> {
        &self.gl_send
    }
}
impl CanMove for Player {
    fn set_pos(&mut self, pos: [i32; 2]) {
        self.position = pos;
    }

    fn gl_mut(&mut self) -> &mut Object {
        &mut self.gl
    }
}
impl Entity for Crate {
    fn position(&self) -> [i32; 2] {
        self.position
    }
    fn gl(&self) -> Object {
        self.gl.clone()
    }
    fn gl_send(&self) -> &Sender<GLFunc> {
        &self.gl_send
    }
}
impl CanMove for Crate {
    fn set_pos(&mut self, pos: [i32; 2]) {
        self.position = pos;
    }
    fn gl_mut(&mut self) -> &mut Object {
        &mut self.gl
    }
}
impl Entity for FloorMarker {
    fn position(&self) -> [i32; 2] {
        self.position
    }
    fn gl(&self) -> Object {
        self.gl.clone()
    }
    fn gl_send(&self) -> &Sender<GLFunc> {
        &self.gl_send
    }
}
impl CanMove for FloorMarker {
    fn set_pos(&mut self, pos: [i32; 2]) {
        self.position = pos;
    }
    fn gl_mut(&mut self) -> &mut Object {
        &mut self.gl
    }
}
impl CanMove for Wall {
    fn set_pos(&mut self, pos: [i32; 2]) {
        self.position = pos;
    }
    fn gl_mut(&mut self) -> &mut Object {
        &mut self.gl
    }
}
trait Reset: CanMove {
    fn reset(&mut self, object: &Object) {
        self.set_pos([0; 2]);
        self.gl_mut().vertices = object.vertices.clone();
        self.update_self();
    }
}
impl Reset for Player {}
impl Reset for FloorMarker {}
impl Reset for Wall {}
impl Reset for Crate {}
///Struct for a Player
struct Player {
    gl: Object,
    position: [i32; 2],
    gl_send: Sender<GLFunc>,
}
///Struct for a Floor Tile
struct FloorMarker {
    gl: Object,
    position: [i32; 2],
    gl_send: Sender<GLFunc>,
}
///Struct for a Wall
struct Wall {
    gl: Object,
    position: [i32; 2],
    is_active: bool,
    gl_send: Sender<GLFunc>,
}
///Struct for a Crate
struct Crate {
    gl: Object,
    position: [i32; 2],
    ///Function to determine if this crate is in the correct position
    correct: fn(&Crate, &[FloorMarker]) -> bool,
    ///Function to determine if this crate is in a bad position,
    ///this returns two bools, the first of which says this crate is stuck
    ///and the second saying the crate is permanantly stuck and the game's
    ///unwinnable
    screwed: fn(&Crate, &[Wall], &[Crate]) -> (bool, bool),
    gl_send: Sender<GLFunc>,
}
///Common behaviours for all objects in the game.
///This allows interfacing with them abstract of their specific types.
///Methods can instead be generic over Entity rather than the specific
///structures
trait Entity {
    ///Returns the position of this object
    fn position(&self) -> [i32; 2];
    ///Returns the GL element of this object
    fn gl(&self) -> Object;
    ///Returns a reference to the GL sender this object owns
    fn gl_send(&self) -> &Sender<GLFunc>;
    ///If this object is currently active
    fn is_active(&self) -> bool {
        true
    }
    ///Send a command to the OpenGL thread to refresh the data for this object
    fn update_self(&self) {
        self.gl_send()
            .send(Box::new({
                let gl = self.gl();
                move |device, _| unsafe {
                    gl.update_self(&device);
                }
            }))
            .unwrap();
    }
}

impl Entity for Wall {
    fn position(&self) -> [i32; 2] {
        self.position
    }
    fn gl(&self) -> Object {
        self.gl.clone()
    }
    fn gl_send(&self) -> &Sender<GLFunc> {
        &self.gl_send
    }
    fn is_active(&self) -> bool {
        self.is_active
    }
}
///This trait is for objects that can move,
///which is actually everything at the moment due to the way I've done level loading.
///This trait inherits from the Entity trait, meaning only types that implement Entity can implement
///CanMove
trait CanMove: Entity {
    ///Change the position of the object, to be used in combo with Entity's get_pos()
    fn set_pos(&mut self, pos: [i32; 2]);
    ///Change the GL data of the object, to be used in combo with Entity's gl()
    fn gl_mut(&mut self) -> &mut Object;
    ///Adjusts the GL data and game position data from input game coords
    fn move_obj(&mut self, x: i32, y: i32) {
        let mut pos = self.position();
        pos[0] += x;
        pos[1] += y;
        self.gl_mut().translate(2. * x as f32, 0., 2. * y as f32);
        self.update_self();
        self.set_pos(pos);
    }
}
///This structure holds the Events system. This could probably be rolled into Game now.
///However this was originally on a different thread, but for some reason that broke the game on Windows /shrug
struct Events {
    camera_sender: Sender<CamFunc>,
    gl_sender: Sender<GLFunc>,
    game_sender: Sender<GameFunc>,
    events_loop: glutin::EventsLoop,
}
impl Events {
    fn new(
        cam: Sender<CamFunc>,
        gl_sender: Sender<GLFunc>,
        events_loop: glutin::EventsLoop,
        game_sender: Sender<GameFunc>,
    ) -> Self {
        Self {
            camera_sender: cam,
            gl_sender,
            events_loop,
            game_sender,
        }
    }
    ///Handles keyboard inputs
    fn poll(&mut self) {
        let mut for_gl: Vec<GLFunc> = Vec::new();
        let mut for_camera: Vec<CamFunc> = Vec::new();
        let mut for_game: Vec<GameFunc> = Vec::new();

        self.events_loop.poll_events(|event| {
            use glutin::{ElementState, Event, WindowEvent};
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput { input, .. } => {
                        if input.state == ElementState::Pressed {
                            if let Some(keycode) = input.virtual_keycode {
                                match keycode {
                                    VirtualKeyCode::W => for_camera.push(Box::new({
                                        move |camera| {
                                            camera.translate(0., 0., 0.05);
                                        }
                                    })),
                                    VirtualKeyCode::A => for_camera.push(Box::new({
                                        move |camera| {
                                            camera.translate(0.05, 0., 0.);
                                        }
                                    })),
                                    VirtualKeyCode::E => for_camera.push(Box::new({
                                        move |camera| {
                                            camera.translate(0., -0.05, 0.00);
                                        }
                                    })),
                                    VirtualKeyCode::Q => for_camera.push(Box::new({
                                        move |camera| {
                                            camera.translate(0.00, 0.05, 0.);
                                        }
                                    })),

                                    VirtualKeyCode::S => for_camera.push(Box::new({
                                        move |camera| {
                                            camera.translate(0., 0., -0.05);
                                        }
                                    })),
                                    VirtualKeyCode::D => for_camera.push(Box::new({
                                        move |camera| {
                                            camera.translate(-0.05, 0., 0.0);
                                        }
                                    })),
                                    VirtualKeyCode::Up => for_game.push(Box::new({
                                        move |game| game.move_player(Direction::Up)
                                    })),
                                    VirtualKeyCode::Down => for_game.push(Box::new({
                                        move |game| game.move_player(Direction::Down)
                                    })),
                                    VirtualKeyCode::Left => for_game.push(Box::new({
                                        move |game| game.move_player(Direction::Left)
                                    })),
                                    VirtualKeyCode::Right => for_game.push(Box::new({
                                        move |game| game.move_player(Direction::Right)
                                    })),
                                    _ => (),
                                }
                            }
                        }
                    }
                    glutin::WindowEvent::Resized(w, h) => for_gl.push(Box::new({
                        move |device, window| unsafe {
                            let perspective =
                                cgmath::perspective(Deg(68.), w as f32 / h as f32, 1.0, 5000.0);
                            gl::UniformMatrix4fv(
                                gl::GetUniformLocation(
                                    device.shader_programme,
                                    CString::new("u_Proj").unwrap().as_ptr(),
                                ),
                                1,
                                gl::FALSE,
                                perspective.as_data_ptr(),
                            );

                            window.resize(w, h);
                            window.context().resize(w, h);
                        }
                    })),
                    glutin::WindowEvent::Closed => {
                        for_game.push(Box::new({ move |game| game.running = false }));
                    }

                    _ => (),
                },
                _ => (),
            }
        });
        while let Some(func) = for_gl.pop() {
            self.gl_sender.send(func).unwrap();
        }
        while let Some(func) = for_camera.pop() {
            self.camera_sender.send(func).unwrap();
        }
        while let Some(func) = for_game.pop() {
            self.game_sender.send(func).unwrap();
        }
    }
}

///Creates a vector of objects from a vector of Vertex and a Vector of indices
fn gen_objects(data: Vec<(Vec<Vertex>, Vec<u32>)>) -> Vec<Object> {
    let mut obj_vec = Vec::new();
    let mut ebo_offset = 0;
    let mut index_offset = 0;
    let mut vbo_offset = 0;
    for i in data {
        obj_vec.push({
            let object = Object {
                ebo_offset: { ebo_offset },
                vbo_offset: { vbo_offset },
                vertices: i.0,
                indices: {
                    let mut indices = i.1.clone();
                    indices.iter_mut().for_each(|x| *x += index_offset);
                    indices
                },
            };

            ebo_offset += (object.indices.len() * 4) as isize;
            vbo_offset += (object.vertices.len() * size_of::<Vertex>()) as isize;
            index_offset += i.1.iter().max().unwrap() + 1;
            object
        });
    }
    obj_vec
}

///This structure holds the graphical data for an entity
#[derive(Debug, Clone)]
struct Object {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    vbo_offset: isize,
    ebo_offset: isize,
}
///This is a general trait for moving objects on the GL side only
trait GLMove {
    fn translate(&mut self, x: f32, y: f32, z: f32);
}

impl Object {
    ///Updates this object, this can only be called on the OpenGL thread or the GPU drivers will throw a fit.
    ///This doesnt need the GLDevice reference to work, but it ensures the command is executed on the same thread
    unsafe fn update_self(&self, _: &GLDevice) {
        gl::BufferSubData(
            gl::ARRAY_BUFFER,
            self.vbo_offset,
            (self.vertices.len() * size_of::<Vertex>()) as isize,
            self.vertices.as_data_ptr(),
        );
        gl::BufferSubData(
            gl::ELEMENT_ARRAY_BUFFER,
            self.ebo_offset,
            (self.indices.len() * 4) as isize,
            self.indices.as_data_ptr(),
        );
    }
}

impl GLMove for Object {
    fn translate(&mut self, x: f32, y: f32, z: f32) {
        self.vertices.iter_mut().for_each(|v| {
            v.pos[0] += x;
            v.pos[1] += y;
            v.pos[2] += z;
        });
    }
}
///Loads a texture onto the GPU
unsafe fn load_texture(img: &image::DynamicImage, buffer: u32) {
    gl::BindTexture(gl::TEXTURE_2D, buffer);
    use image::GenericImage;
    let (x, y) = img.dimensions();
    let (x, y) = (x as i32, y as i32);
    let raw_pixels = img.raw_pixels();
    gl::TexImage2D(
        gl::TEXTURE_2D,
        0,
        gl::RGB8 as i32,
        x,
        y,
        0,
        gl::RGB,
        gl::UNSIGNED_BYTE,
        &raw_pixels[0] as *const u8 as _,
    );
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
}
///Main function
fn main() {
    let events_loop = glutin::EventsLoop::new();
    let builder = glutin::WindowBuilder::new().with_title("Sokoban");
    let context = glutin::ContextBuilder::new().with_multisampling(2);
    let (gl_sender, gl_receiver): (Sender<GLFunc>, Receiver<GLFunc>) = channel();
    let (game_sender, game_receiver): (Sender<GameFunc>, Receiver<GameFunc>) = channel();
    let (camera_sender, camera_receiver): (Sender<CamFunc>, Receiver<CamFunc>) = channel();
    {
        let player_skin = image::open(&Path::new("player.jpg")).unwrap();
        let crate_skin = image::open(&Path::new("container.jpg")).unwrap();
        let floor = image::open(&Path::new("floor.jpg")).unwrap();
        let crate_goes_here_skin = image::open(&Path::new("crate_goes_here.jpg")).unwrap();
        let wall_skin = image::open(&Path::new("wall.jpg")).unwrap();

        thread::spawn({
            let mut window = Window {
                update: false,
                level: 0,
                moves: 0,
                window: glutin::GlWindow::new(builder, context, &events_loop).unwrap(),
            };
            move || {
                let mut device = GLDevice {
                    vao: 0,
                    vbo: 0,
                    texture: [0; 5],
                    ebo: 0,
                    shader_programme: 0,
                };
                unsafe {
                    window.make_current().unwrap();
                    gl::load_with(|s| window.get_proc_address(s) as *const _);
                }
                loop {
                    while let Ok(func) = gl_receiver.try_recv() {
                        func(&mut device, &mut window);
                    }
                    unsafe {
                        gl::Clear(gl::COLOR_BUFFER_BIT);
                        gl::Clear(gl::DEPTH_BUFFER_BIT);

                        gl::BindTexture(gl::TEXTURE_2D, device.texture[0]);
                        gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());
                        gl::BindTexture(gl::TEXTURE_2D, device.texture[1]);
                        gl::DrawElements(
                            gl::TRIANGLES,
                            2904,
                            gl::UNSIGNED_INT,
                            (6 * size_of::<GLuint>()) as *const c_void,
                        );
                        gl::BindTexture(gl::TEXTURE_2D, device.texture[2]);
                        gl::DrawElements(
                            gl::TRIANGLES,
                            2 * 36,
                            gl::UNSIGNED_INT,
                            ((2904 + 6) * size_of::<GLuint>()) as *const c_void,
                        );
                        gl::BindTexture(gl::TEXTURE_2D, device.texture[3]);
                        gl::DrawElements(
                            gl::TRIANGLES,
                            2 * 6,
                            gl::UNSIGNED_INT,
                            ((2904 + 6 + (2 * 36)) * size_of::<GLuint>()) as *const c_void,
                        );
                        gl::BindTexture(gl::TEXTURE_2D, device.texture[4]);
                        gl::DrawElements(
                            gl::TRIANGLES,
                            43 * 36,
                            gl::UNSIGNED_INT,
                            ((2904 + 6 + 2 * 36 + (2 * 6)) * size_of::<GLuint>()) as *const c_void,
                        );
                    }
                    if window.update {
                        window.update_window();
                    }
                    window.update = false;
                    window.swap_buffers().unwrap();
                    sleep(Duration::from_millis(WAIT_TIME));
                }
            }
        });

        gl_sender
            .send(Box::new(move |device, _| unsafe {
                gl::ClearColor(0.0, 0.0, 0.0, 1.0);
                gl::Enable(gl::DEPTH_TEST);
                gl::Enable(gl::CULL_FACE);
                gl::Enable(gl::MULTISAMPLE);
                gl::CullFace(gl::BACK);
                gl::FrontFace(gl::CCW);
                gl::GenVertexArrays(1, &mut device.vao);
                gl::BindVertexArray(device.vao);
                gl::GenBuffers(1, &mut device.vbo);
                gl::BindBuffer(gl::ARRAY_BUFFER, device.vbo);
                {
                    let initialisation_buffer = vec![-10_f32; size_of::<Vertex>() * 3_058];
                    gl::BufferData(
                        gl::ARRAY_BUFFER,
                        size_of::<Vertex>() as isize * 3_058,
                        &initialisation_buffer[0] as *const f32 as *const std::os::raw::c_void,
                        gl::STATIC_DRAW,
                    );
                }
                device.shader_programme = gen_shader_programme(
                    &include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/vertex.vert"))[..],
                    &include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/fragment.frag"))[..],
                );
                gl::UseProgram(device.shader_programme);
                gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 32, std::ptr::null());
                gl::VertexAttribPointer(1, 3, gl::FLOAT, gl::FALSE, 32, 12 as *const c_void);
                gl::VertexAttribPointer(2, 2, gl::FLOAT, gl::FALSE, 32, 24 as *const c_void);
                gl::EnableVertexAttribArray(0);
                gl::EnableVertexAttribArray(1);
                gl::EnableVertexAttribArray(2);
                gl::GenBuffers(1, &mut device.ebo);
                gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, device.ebo);
                let temp = vec![0_u32; 4542];
                gl::BufferData(
                    gl::ELEMENT_ARRAY_BUFFER,
                    4542 * 4,
                    &temp[0] as *const u32 as *const _,
                    gl::STATIC_DRAW,
                );
                gl::GenTextures(5, &mut device.texture[0]);
                load_texture(&floor, device.texture[0]);
                load_texture(&player_skin, device.texture[1]);
                load_texture(&crate_skin, device.texture[2]);
                load_texture(&crate_goes_here_skin, device.texture[3]);
                load_texture(&wall_skin, device.texture[4]);

                gl::Uniform3f(
                    gl::GetUniformLocation(
                        device.shader_programme,
                        CString::new("light_pos").unwrap().as_ptr(),
                    ),
                    0.,
                    20.,
                    0.,
                );
                gl::Uniform1f(
                    gl::GetUniformLocation(
                        device.shader_programme,
                        CString::new("ambient_strength").unwrap().as_ptr(),
                    ),
                    0.7,
                );
                gl::Uniform1i(
                    gl::GetUniformLocation(
                        device.shader_programme,
                        CString::new("spec_samples").unwrap().as_ptr(),
                    ),
                    2,
                );
            }))
            .unwrap();
    }
    let mut game = {
        let (model, _) = tobj::load_obj(&Path::new("cube.obj")).unwrap();
        let (monkey, _) = tobj::load_obj(&Path::new("monkey.obj")).unwrap();
        let (mut crate_vertices, crate_indices) = model[0].from_obj_model();
        crate_vertices
            .iter_mut()
            .for_each(|z| z.uv[1] = 1. - z.uv[1]);
        let (mut monkey_vertices, monkey_indices) = monkey[0].from_obj_model();
        monkey_vertices.iter_mut().for_each(|z| {
            z.uv[1] = 1. - z.uv[1];
            z.pos[0] += -1.0;
        });
        let mut in_vec = Vec::new();
        let floor = vec![
            Vertex {
                pos: [-20.0, -1.0, -20.0],
                uv: [0., 1.],
                ..Default::default()
            },
            Vertex {
                pos: [20.0, -1.0, 20.0],

                uv: [1., 0.],
                ..Default::default()
            },
            Vertex {
                pos: [-20.0, -1.0, 20.0],
                uv: [0., 0.],
                ..Default::default()
            },
            Vertex {
                pos: [20.0, -1.0, -20.0],
                uv: [1., 1.],
                ..Default::default()
            },
        ];
        let box_dest = vec![
            Vertex {
                pos: [-1.0, -0.95, -1.0],
                uv: [0., 1.],
                ..Default::default()
            },
            Vertex {
                pos: [1.0, -0.95, 1.0],
                uv: [1., 0.],
                ..Default::default()
            },
            Vertex {
                pos: [-1.0, -0.95, 1.0],
                uv: [0., 0.],
                ..Default::default()
            },
            Vertex {
                pos: [1.0, -0.95, -1.0],
                uv: [1., 1.],
                ..Default::default()
            },
        ];
        let floor_indices = vec![2, 1, 0, 1, 3, 0];
        let mut wall_positions: Vec<[i32; 2]> = Vec::new();
        for _ in 0..10 {
            wall_positions.push([-4, -4]);
        }
        wall_positions.extend((-4..4).map(|x| [-4, x]).collect::<Vec<[i32; 2]>>());
        wall_positions.extend((-4..4).map(|x| [4, x]).collect::<Vec<[i32; 2]>>());
        wall_positions.extend((-4..4).map(|x| [x, -4]).collect::<Vec<[i32; 2]>>());
        wall_positions.extend((-4..5).map(|x| [x, 4]).collect::<Vec<[i32; 2]>>());
        in_vec.push((floor.clone(), floor_indices.clone()));
        in_vec.push((monkey_vertices.clone(), monkey_indices.clone()));
        for _ in 0..2 {
            in_vec.push((crate_vertices.clone(), crate_indices.clone()));
        }
        for _ in 0..2 {
            in_vec.push((box_dest.clone(), floor_indices.clone()));
        }
        for _ in 0..wall_positions.len() {
            in_vec.push((crate_vertices.clone(), crate_indices.clone()));
        }
        let mut objects = gen_objects(in_vec);
        gl_sender
            .send(Box::new({
                let objects = objects.clone();
                move |device, _| unsafe {
                    for i in objects.clone().iter() {
                        i.update_self(&device);
                    }
                }
            }))
            .unwrap();
        let player = Player {
            position: [0; 2],
            gl: objects[1].clone(),
            gl_send: gl_sender.clone(),
        };
        let mut crates = objects
            .iter()
            .skip(2)
            .take(2)
            .map(|x| Crate {
                gl: x.clone(),
                gl_send: gl_sender.clone(),
                position: [0; 2],
                correct: |crat, floors: &[FloorMarker]| {
                    floors.iter().any(|x| crat.position == x.position)
                },
                screwed: |crat: &Crate, walls: &[Wall], crates: &[Crate]| {
                    let my_position = crat.position();
                    let (mut x_match, mut y_match) = (false, false);
                    for i in walls {
                        if i.is_active {
                            let wall_position = i.position();
                            if my_position[0] == wall_position[0] {
                                if my_position[1] - wall_position[1] == -1
                                    || my_position[1] - wall_position[1] == 1
                                {
                                    x_match = true;
                                }
                            }
                            if my_position[1] == wall_position[1] {
                                if my_position[0] - wall_position[0] == -1
                                    || my_position[0] - wall_position[0] == 1
                                {
                                    y_match = true;
                                }
                            }
                        }
                    }
                    if x_match && y_match {
                        return (true, true);
                    }
                    for i in crates {
                        let wall_position = i.position();
                        if my_position[0] == wall_position[0] {
                            if my_position[1] - wall_position[1] == -1
                                || my_position[1] - wall_position[1] == 1
                            {
                                x_match = true;
                            }
                        }
                        if my_position[1] == wall_position[1] {
                            if my_position[0] - wall_position[0] == -1
                                || my_position[0] - wall_position[0] == 1
                            {
                                y_match = true;
                            }
                        }
                    }
                    (x_match && y_match, false)
                },
            })
            .collect::<Vec<_>>();
        let mut floor_tiles = objects
            .iter()
            .skip(4)
            .take(2)
            .map(|x| FloorMarker {
                gl: x.clone(),
                gl_send: gl_sender.clone(),
                position: [0; 2],
            })
            .collect::<Vec<_>>();

        let mut walls = objects
            .iter()
            .skip(6)
            .map(|x| Wall {
                gl: x.clone(),
                gl_send: gl_sender.clone(),
                is_active: true,
                position: [0; 2],
            })
            .collect::<Vec<Wall>>();
        for (wall, wall_pos) in walls.iter_mut().zip(wall_positions.iter()) {
            wall.move_obj(wall_pos[0], wall_pos[1]);
        }

        let mut game = Game::new(
            vec![
                objects[0].clone(),
                objects[1].clone(),
                objects[2].clone(),
                objects[4].clone(),
                objects[6].clone(),
            ],
            player,
            gl_sender.clone(),
            camera_sender.clone(),
            game_receiver,
        );
        game.crates = crates;
        game.walls = walls;
        for i in &floor_tiles {
            i.update_self();
        }
        game.floor_tiles = floor_tiles;
        game.levels.push(level_zero);
        game.levels.push(level_one);
        game.levels.push(level_two);
        game.levels.push(level_three);
        game.levels.push(level_four);
        game.subroutine.insert(3, Arc::new(RwLock::new(false)));
        game.subroutine.insert(4, Arc::new(RwLock::new(false)));
        game
    };
    thread::spawn({
        let gl_sender = gl_sender.clone();
        move || {
            let mut camera = Camera::new(camera_receiver, gl_sender);
            camera.send_view();
            camera.run();
        }
    });

    let mut events = Events::new(
        camera_sender.clone(),
        gl_sender.clone(),
        events_loop,
        game_sender.clone(),
    );
    (game.levels[0])(&mut game);
    game.run(&mut events);
}
