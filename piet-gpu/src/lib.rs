mod pico_svg;
mod render_ctx;

pub use render_ctx::PietGpuRenderContext;

use bytemuck::{Pod, Zeroable};
use pico_svg::PicoSvg;
use piet::kurbo::{BezPath, Circle, Line, Point, Vec2};
use piet::{Color, RenderContext};
use piet_gpu_types::encoder::Encode;
use std::{iter, mem};
use wgpu::util::DeviceExt as _;

pub const WIDTH: usize = TILE_W * WIDTH_IN_TILES;
pub const HEIGHT: usize = TILE_H * HEIGHT_IN_TILES;

const TILE_W: usize = 16;
const TILE_H: usize = 16;

const WIDTH_IN_TILES: usize = 128;
const HEIGHT_IN_TILES: usize = 96;
const PTCL_INITIAL_ALLOC: usize = 1024;

pub fn render_svg(rc: &mut impl RenderContext, filename: &str, scale: f64) {
    let xml_str = std::fs::read_to_string(filename).unwrap();
    let start = std::time::Instant::now();
    let svg = PicoSvg::load(&xml_str, scale).unwrap();
    println!("parsing time: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    svg.render(rc);
    println!("flattening and encoding time: {:?}", start.elapsed());
}

pub fn render_scene(rc: &mut impl RenderContext) {
    render_tiger(rc);
}

fn render_tiger(rc: &mut impl RenderContext) {
    let xml_str = std::str::from_utf8(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/Ghostscript_Tiger.svg"
    )))
    .unwrap();
    let start = std::time::Instant::now();
    let svg = PicoSvg::load(xml_str, 8.0).unwrap();
    println!("parsing time: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    svg.render(rc);
    println!("flattening and encoding time: {:?}", start.elapsed());
}

pub struct Renderer {
    scene_buf: wgpu::Buffer,
    scene_dev: wgpu::Buffer,

    scene_size: u64,

    pub output_texture: wgpu::Texture,

    zeroed_state_buf: wgpu::Buffer,
    pub state_buf: wgpu::Buffer,
    pub anno_buf: wgpu::Buffer,
    pub pathseg_buf: wgpu::Buffer,
    pub tile_buf: wgpu::Buffer,
    pub bin_buf: wgpu::Buffer,
    pub ptcl_buf: wgpu::Buffer,

    render_pipeline: RenderPipeline,

    el_pipeline: ComputePipeline,

    tile_pipeline: ComputePipeline,

    path_pipeline: ComputePipeline,

    backdrop_pipeline: ComputePipeline,

    tile_alloc_buf_host: wgpu::Buffer,
    tile_alloc_buf_dev: wgpu::Buffer,

    bin_pipeline: ComputePipeline,

    bin_alloc_buf_host: wgpu::Buffer,
    bin_alloc_buf_dev: wgpu::Buffer,

    coarse_pipeline: ComputePipeline,

    coarse_alloc_buf_host: wgpu::Buffer,
    coarse_alloc_buf_dev: wgpu::Buffer,

    k4_pipeline: ComputePipeline,

    anno_bind_group: wgpu::BindGroup,

    n_elements: usize,
    n_paths: usize,
    n_pathseg: usize,
}

struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

struct RenderPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
}

#[derive(Debug, Clone, Copy)]
struct TexturedVertex {
    pos: [f32; 4],
    tex_coord: [f32; 2],
}

unsafe impl Pod for TexturedVertex {}
unsafe impl Zeroable for TexturedVertex {}

impl RenderPipeline {
    fn new(
        device: &wgpu::Device,
        vertex_spirv: &[u8],
        fragment_spirv: &[u8],
        input: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> Self {
        use memoffset::offset_of;

        let vs_module = device.create_shader_module(wgpu::util::make_spirv(vertex_spirv));
        let fs_module = device.create_shader_module(wgpu::util::make_spirv(fragment_spirv));

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[
                TexturedVertex {
                    pos: [-1., -1., 0., 1.],
                    tex_coord: [0., 1.],
                },
                TexturedVertex {
                    pos: [1., -1., 0., 1.],
                    tex_coord: [1., 1.],
                },
                TexturedVertex {
                    pos: [1., 1., 0., 1.],
                    tex_coord: [1., 0.],
                },
                TexturedVertex {
                    pos: [1., 1., 0., 1.],
                    tex_coord: [1., 0.],
                },
                TexturedVertex {
                    pos: [-1., 1., 0., 1.],
                    tex_coord: [0., 0.],
                },
                TexturedVertex {
                    pos: [-1., -1., 0., 1.],
                    tex_coord: [0., 1.],
                },
            ]),
            usage: wgpu::BufferUsage::VERTEX,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::default(),
                cull_mode: wgpu::CullMode::default(),
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float4,
                            offset: offset_of!(TexturedVertex, pos) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float2,
                            offset: offset_of!(TexturedVertex, tex_coord) as u64,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        RenderPipeline {
            pipeline,
            bind_group,
            vertex_buffer,
        }
    }

    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, attachment: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..6, 0..1);
    }
}

impl ComputePipeline {
    fn new(
        device: &wgpu::Device,
        spirv: &[u8],
        buffers: &[&wgpu::Buffer],
        images: &[&wgpu::TextureView],
        extra_layout: Option<&wgpu::BindGroupLayout>,
    ) -> Self {
        let entries = (0..buffers.len())
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .chain((0..images.len()).map(|i| wgpu::BindGroupLayoutEntry {
                binding: (i + buffers.len()) as u32,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    readonly: false,
                    dimension: wgpu::TextureViewDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                },
                count: None,
            }))
            .collect::<Vec<_>>();

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let bind_group_layout = &bind_group_layout;

        // Workaround for lifetime issues
        let bgl0;
        let bgl1;

        let bind_group_layouts: &[_] = if let Some(extra_layout) = extra_layout {
            bgl0 = [bind_group_layout, extra_layout];
            &bgl0
        } else {
            bgl1 = [bind_group_layout];
            &bgl1
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout_world"),
            bind_group_layouts,
            push_constant_ranges: &[],
        });

        // Create compute pipeline.
        let compute_shader_module = device.create_shader_module(wgpu::util::make_spirv(spirv));

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: &compute_shader_module,
                entry_point: "main",
            },
        });

        let entries = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::Buffer(buf.slice(..)),
            })
            .chain(
                images
                    .iter()
                    .enumerate()
                    .map(|(i, tex)| wgpu::BindGroupEntry {
                        binding: (i + buffers.len()) as u32,
                        resource: wgpu::BindingResource::TextureView(tex),
                    }),
            )
            .collect::<Vec<_>>();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &entries,
        });

        ComputePipeline {
            pipeline,
            bind_group,
        }
    }

    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x: u32,
        y: u32,
        additional: &[&wgpu::BindGroup],
    ) {
        let mut pass = encoder.begin_compute_pass();

        pass.set_pipeline(&self.pipeline);

        for (i, bind_group) in iter::once(&self.bind_group)
            .chain(additional.into_iter().copied())
            .enumerate()
        {
            pass.set_bind_group(i as u32, bind_group, &[]);
        }

        pass.dispatch(x, y, 1);
    }
}

const STATE_SIZE: u64 = (1 * 1024 * 1024) / (1 << 8);
const ANNO_SIZE: u64 = (64 * 1024 * 1024) / (1 << 14);
const PATHSEG_SIZE: u64 = (64 * 1024 * 1024) / (1 << 9);
const TILE_SIZE: u64 = (64 * 1024 * 1024) / (1 << 5);
const BIN_SIZE: u64 = (64 * 1024 * 1024) / (1 << 12);
const PTCL_SIZE: u64 = (48 * 1024 * 1024) / (1 << 2);
const TILE_ALLOC_SIZE: u64 = 12;
const COARSE_ALLOC_SIZE: u64 = 8;
const BIN_ALLOC_SIZE: u64 = 8;

impl Renderer {
    pub fn new(device: &wgpu::Device, scene: &[u8], n_paths: usize, n_pathseg: usize) -> Self {
        let host = wgpu::BufferUsage::COPY_SRC;
        let dev = wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST;

        let n_elements = scene.len() / piet_gpu_types::scene::Element::fixed_size();
        println!(
            "scene: {} elements, {} paths, {} path_segments",
            n_elements, n_paths, n_pathseg
        );

        let scene_size = scene.len() as u64;

        let scene_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: scene,
            usage: host,
        });
        let scene_dev = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: scene_size,
            usage: dev,
            mapped_at_creation: false,
        });

        let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: STATE_SIZE,
            usage: dev,
            mapped_at_creation: false,
        });
        let zeroed_state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0; STATE_SIZE as usize],
            usage: host,
        });
        let anno_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ANNO_SIZE,
            usage: dev,
            mapped_at_creation: false,
        });
        let pathseg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: PATHSEG_SIZE,
            usage: dev,
            mapped_at_creation: false,
        });
        let tile_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: TILE_SIZE,
            usage: dev,
            mapped_at_creation: false,
        });
        let bin_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: BIN_SIZE,
            usage: dev,
            mapped_at_creation: false,
        });
        let ptcl_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: PTCL_SIZE,
            usage: dev,
            mapped_at_creation: false,
        });

        let anno_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let anno_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &anno_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(anno_buf.slice(..)),
            }],
        });

        let el_code = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader/elements.spv"));
        let el_pipeline = ComputePipeline::new(
            &device,
            el_code,
            &[&scene_dev, &state_buf, &pathseg_buf],
            &[],
            Some(&anno_bind_group_layout),
        );

        // TODO: constants
        const PATH_SIZE: usize = 12;
        let tile_alloc_start = ((n_paths + 31) & !31) * PATH_SIZE;

        let tile_alloc_buf_host = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[
                n_paths as u32,
                n_pathseg as u32,
                tile_alloc_start as u32,
            ]),
            usage: host,
        });
        let tile_alloc_buf_dev = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: TILE_ALLOC_SIZE,
            usage: dev,
            mapped_at_creation: false,
        });

        let tile_alloc_code = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shader/tile_alloc.spv"
        ));
        let tile_pipeline = ComputePipeline::new(
            &device,
            tile_alloc_code,
            &[&tile_alloc_buf_dev, &tile_buf],
            &[],
            Some(&anno_bind_group_layout),
        );

        let path_alloc_code = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shader/path_coarse.spv"
        ));
        let path_pipeline = ComputePipeline::new(
            &device,
            path_alloc_code,
            &[&pathseg_buf, &tile_alloc_buf_dev, &tile_buf],
            &[],
            None,
        );

        let backdrop_alloc_code =
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader/backdrop.spv"));
        let backdrop_pipeline = ComputePipeline::new(
            &device,
            backdrop_alloc_code,
            &[&tile_alloc_buf_dev, &tile_buf],
            &[],
            Some(&anno_bind_group_layout),
        );

        let bin_alloc_buf_host = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: BIN_ALLOC_SIZE,
            usage: host,
            mapped_at_creation: true,
        });
        let bin_alloc_buf_dev = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: BIN_ALLOC_SIZE,
            usage: dev,
            mapped_at_creation: false,
        });

        // TODO: constants
        let bin_alloc_start = ((n_paths + 255) & !255) * 8;
        bin_alloc_buf_host
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&[
                n_paths as u32,
                bin_alloc_start as u32,
            ]));
        bin_alloc_buf_host.unmap();

        let bin_code = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader/binning.spv"));
        let bin_pipeline = ComputePipeline::new(
            &device,
            bin_code,
            &[&bin_alloc_buf_dev, &bin_buf],
            &[],
            Some(&anno_bind_group_layout),
        );

        let coarse_alloc_buf_host = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: COARSE_ALLOC_SIZE,
            usage: host,
            mapped_at_creation: true,
        });
        let coarse_alloc_buf_dev = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: COARSE_ALLOC_SIZE,
            usage: dev,
            mapped_at_creation: false,
        });

        let coarse_alloc_start = WIDTH_IN_TILES * HEIGHT_IN_TILES * PTCL_INITIAL_ALLOC;
        coarse_alloc_buf_host
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&[
                n_paths as u32,
                coarse_alloc_start as u32,
            ]));
        coarse_alloc_buf_host.unmap();
        let coarse_code = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader/coarse.spv"));
        let coarse_pipeline = ComputePipeline::new(
            &device,
            coarse_code,
            &[&bin_buf, &tile_buf, &coarse_alloc_buf_dev, &ptcl_buf],
            &[],
            Some(&anno_bind_group_layout),
        );

        let output_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: None,
        });

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: WIDTH as u32,
                height: HEIGHT as u32,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::STORAGE
                | wgpu::TextureUsage::COPY_SRC,
        });

        let output_texture_view = output_texture.create_view(&Default::default());

        let k4_code = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader/kernel4.spv"));
        let k4_pipeline = ComputePipeline::new(
            &device,
            k4_code,
            &[&ptcl_buf, &tile_buf],
            &[&output_texture_view],
            None,
        );

        let vs_code = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shader/render.vert.spv"
        ));
        let fs_code = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shader/render.frag.spv"
        ));
        let render_pipeline = RenderPipeline::new(
            device,
            vs_code,
            fs_code,
            &output_texture_view,
            &output_sampler,
        );

        Renderer {
            scene_buf,
            scene_dev,
            scene_size,

            output_texture,

            render_pipeline,

            el_pipeline,

            tile_pipeline,

            path_pipeline,

            backdrop_pipeline,

            bin_pipeline,

            coarse_pipeline,

            k4_pipeline,

            zeroed_state_buf,
            state_buf,
            anno_buf,
            pathseg_buf,
            tile_buf,
            bin_buf,
            ptcl_buf,

            tile_alloc_buf_host,
            tile_alloc_buf_dev,
            bin_alloc_buf_host,
            bin_alloc_buf_dev,
            coarse_alloc_buf_host,
            coarse_alloc_buf_dev,

            anno_bind_group,

            n_elements,
            n_paths,
            n_pathseg,
        }
    }

    pub fn record(&self, cmd_buf: &mut wgpu::CommandEncoder, output: Option<&wgpu::TextureView>) {
        cmd_buf.copy_buffer_to_buffer(&self.scene_buf, 0, &self.scene_dev, 0, self.scene_size);
        cmd_buf.copy_buffer_to_buffer(
            &self.tile_alloc_buf_host,
            0,
            &self.tile_alloc_buf_dev,
            0,
            TILE_ALLOC_SIZE,
        );
        cmd_buf.copy_buffer_to_buffer(
            &self.bin_alloc_buf_host,
            0,
            &self.bin_alloc_buf_dev,
            0,
            BIN_ALLOC_SIZE,
        );
        cmd_buf.copy_buffer_to_buffer(
            &self.coarse_alloc_buf_host,
            0,
            &self.coarse_alloc_buf_dev,
            0,
            COARSE_ALLOC_SIZE,
        );
        cmd_buf.copy_buffer_to_buffer(&self.zeroed_state_buf, 0, &self.state_buf, 0, STATE_SIZE);

        self.el_pipeline.dispatch(
            cmd_buf,
            ((self.n_elements + 127) / 128) as u32,
            1,
            &[&self.anno_bind_group],
        );
        self.tile_pipeline.dispatch(
            cmd_buf,
            ((self.n_paths + 255) / 256) as u32,
            1,
            &[&self.anno_bind_group],
        );
        self.path_pipeline
            .dispatch(cmd_buf, ((self.n_pathseg + 31) / 32) as u32, 1, &[]);
        self.backdrop_pipeline.dispatch(
            cmd_buf,
            ((self.n_paths + 255) / 256) as u32,
            1,
            &[&self.anno_bind_group],
        );
        self.bin_pipeline.dispatch(
            cmd_buf,
            ((self.n_paths + 255) / 256) as u32,
            1,
            &[&self.anno_bind_group],
        );
        self.coarse_pipeline.dispatch(
            cmd_buf,
            WIDTH as u32 / 256,
            HEIGHT as u32 / 256,
            &[&self.anno_bind_group],
        );
        self.k4_pipeline.dispatch(
            cmd_buf,
            (WIDTH / TILE_W) as u32,
            (HEIGHT / TILE_H) as u32,
            &[],
        );

        if let Some(output) = output {
            self.render_pipeline.dispatch(cmd_buf, output);
        }
    }
}
