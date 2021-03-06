use piet_gpu::{render_scene, PietGpuRenderContext, Renderer, HEIGHT, WIDTH};
use std::iter;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop)?;

    Ok(futures::executor::block_on(run(window, event_loop)))
}

async fn run(window: Window, event_loop: EventLoop<()>) {
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
        })
        .await
        .unwrap();

    let out_path = if cfg!(debug_assertions) {
        use std::convert::TryFrom;

        Some(
            std::path::PathBuf::try_from(env!("CARGO_MANIFEST_DIR"))
                .unwrap()
                .join("calls.dbg"),
        )
    } else {
        None
    };

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::default(),
                limits: wgpu::Limits::default(),
                shader_validation: cfg!(debug_assertions),
            },
            out_path.as_ref().map(|p| &**p),
        )
        .await
        .unwrap();

    let size = window.inner_size();
    let sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    let mut swapchain = device.create_swap_chain(&surface, &sc_desc);

    let mut ctx = PietGpuRenderContext::new();
    render_scene(&mut ctx);
    let n_paths = ctx.path_count();
    let n_pathseg = ctx.pathseg_count();
    let scene = ctx.get_scene_buf();

    let renderer = Renderer::new(&device, scene, n_paths, n_pathseg);

    {
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (4 * WIDTH * HEIGHT) as _,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
            mapped_at_creation: false,
        });

        let mut cmd_buf =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        renderer.record(&mut cmd_buf, None);

        queue.submit(iter::once(cmd_buf.finish()));

        let slice = out_buf.slice(..);
        let fut = slice.map_async(wgpu::MapMode::Read);

        device.poll(wgpu::Maintain::Wait);

        fut.await.unwrap();

        image::save_buffer(
            "./out.png",
            &*slice.get_mapped_range(),
            WIDTH as u32,
            HEIGHT as u32,
            image::ColorType::Rgba8,
        )
        .unwrap();

        dbg!();

        out_buf.unmap();
    }

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait; // `ControlFlow::Wait` if only re-render on event

        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::Resized(size) => {
                    let sc_desc = wgpu::SwapChainDescriptor {
                        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        width: size.width,
                        height: size.height,
                        present_mode: wgpu::PresentMode::Mailbox,
                    };

                    swapchain = device.create_swap_chain(&surface, &sc_desc);
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                let swap_frame = swapchain.get_current_frame().unwrap();
                let mut cmd_buf =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                renderer.record(&mut cmd_buf, Some(&swap_frame.output.view));

                queue.submit(iter::once(cmd_buf.finish()));
            }
            _ => (),
        }
    });
}
