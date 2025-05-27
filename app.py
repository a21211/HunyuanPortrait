import os
import gradio as gr
from omegaconf import OmegaConf
from inference import main
import numpy as np
import cv2
import logging
import shutil
import glob

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载配置文件
config_path = "./config/hunyuan-portrait.yaml"
cfg = OmegaConf.load(config_path)

def get_reference_files():
    """
    获取参考图像和视频文件列表
    Returns:
        tuple: (参考图像列表, 参考视频文件名列表, 参考视频路径映射)
    """
    # 获取参考文件目录
    ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "reference")
    os.makedirs(ref_dir, exist_ok=True)
    
    # 获取所有图片和视频文件
    image_files = (glob.glob(os.path.join(ref_dir, "*.png")) + 
                  glob.glob(os.path.join(ref_dir, "*.jpg")) + 
                  glob.glob(os.path.join(ref_dir, "*.jpeg")))
    video_files = (glob.glob(os.path.join(ref_dir, "*.mp4")) + 
                  glob.glob(os.path.join(ref_dir, "*.avi")) + 
                  glob.glob(os.path.join(ref_dir, "*.mov")))
    
    # 按文件名排序
    image_files.sort()
    video_files.sort()
    
    # 转换为预览格式
    image_previews = []
    for img_path in image_files:
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_previews.append((img, os.path.basename(img_path)))
        except Exception as e:
            logger.warning(f"无法加载图片 {img_path}: {str(e)}")
    
    # 只返回文件名列表和路径映射
    video_names = [os.path.basename(v) for v in video_files]
    video_map = {os.path.basename(v): v for v in video_files}
    
    return image_previews, video_names, video_map

def save_temp_file(data, prefix, suffix, progress=None):
    """
    将数据保存为临时文件
    Args:
        data: 图片数据或视频文件路径
        prefix: 文件名前缀
        suffix: 文件后缀
        progress: 进度条对象
    Returns:
        str: 临时文件路径
    """
    # 创建tmp目录
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    temp_path = os.path.join(tmp_dir, f"{prefix}_{os.urandom(4).hex()}{suffix}")
    logger.info(f"保存临时文件: {temp_path}")
    
    try:
        if isinstance(data, np.ndarray):
            if len(data.shape) == 3:  # 图片
                cv2.imwrite(temp_path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
                logger.info(f"图片已保存: {temp_path}")
                if progress is not None:
                    try:
                        progress(0.5, desc="图片保存完成")
                    except Exception as e:
                        logger.warning(f"更新进度条失败: {str(e)}")
            else:  # 视频数据
                # 检查视频数据的格式
                logger.info(f"视频数据形状: {data.shape}")
                logger.info(f"视频数据类型: {data.dtype}")
                logger.info(f"视频数据范围: [{data.min()}, {data.max()}]")
                
                height, width = data[0].shape[:2]
                logger.info(f"视频尺寸: {width}x{height}, 帧数: {len(data)}")
                
                # 使用OpenCV保存视频
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_path, fourcc, 30.0, (width, height))
                
                if not out.isOpened():
                    raise RuntimeError("无法创建视频写入器")
                
                # 写入帧数据
                total_frames = len(data)
                for i, frame in enumerate(data):
                    # 确保帧数据在正确的范围内
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if i % 10 == 0:
                        logger.info(f"处理第 {i+1} 帧")
                        if progress is not None:
                            try:
                                progress((i + 1) / total_frames, 
                                       desc=f"处理视频帧 {i+1}/{total_frames}")
                            except Exception as e:
                                logger.warning(f"更新进度条失败: {str(e)}")
                    success = out.write(frame_bgr)
                    if not success:
                        raise RuntimeError(f"写入第 {i+1} 帧失败")
                
                out.release()
                logger.info("视频处理完成")
                if progress is not None:
                    try:
                        progress(1.0, desc="视频处理完成")
                    except Exception as e:
                        logger.warning(f"更新进度条失败: {str(e)}")
                
        elif isinstance(data, str):  # 视频文件路径
            # 复制视频文件
            shutil.copy2(data, temp_path)
            logger.info(f"视频文件已复制: {temp_path}")
            if progress is not None:
                try:
                    progress(1.0, desc="视频文件复制完成")
                except Exception as e:
                    logger.warning(f"更新进度条失败: {str(e)}")
        else:
            logger.error(f"数据类型错误: {type(data)}")
            raise RuntimeError(f"不支持的数据类型: {type(data)}")
        
        if not os.path.exists(temp_path):
            raise RuntimeError(f"文件保存失败: {temp_path}")
        
        file_size = os.path.getsize(temp_path)
        logger.info(f"文件大小: {file_size} 字节")
        
        if file_size == 0:
            raise RuntimeError(f"文件大小为0: {temp_path}")
        
        return temp_path
        
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def generate_video(image, video, progress=gr.Progress()):
    """
    调用inference.py中的main函数生成动画视频
    Args:
        image: 上传的图片数据
        video: 上传的视频文件路径
        progress: 进度条对象
    Returns:
        str: 生成的视频文件路径
    """
    logger.info("开始生成视频...")
    logger.info(f"视频数据类型: {type(video)}")
    
    if video is None:
        raise RuntimeError("未接收到视频数据")
    
    # 创建输出目录
    output_dir = cfg.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
    
    try:
        # 保存临时文件
        try:
            progress(0.1, desc="保存图片...")
        except Exception as e:
            logger.warning(f"更新进度条失败: {str(e)}")
            
        image_path = save_temp_file(image, "input_image", ".png", progress)
        
        try:
            progress(0.2, desc="保存视频...")
        except Exception as e:
            logger.warning(f"更新进度条失败: {str(e)}")
            
        video_path = save_temp_file(video, "input_video", ".mp4", progress)
        
        # 调用inference.py中的main函数
        args = type('Args', (), {
            'video_path': video_path,
            'image_path': image_path
        })
        logger.info(f"调用main函数，参数: video_path={video_path}, "
                   f"image_path={image_path}")
        
        try:
            progress(0.3, desc="开始生成动画...")
        except Exception as e:
            logger.warning(f"更新进度条失败: {str(e)}")
            
        main(cfg, args)
        
        # 返回生成的视频文件路径
        output_file = os.path.join(output_dir, os.listdir(output_dir)[-1])
        logger.info(f"生成完成，输出文件: {output_file}")
        
        try:
            progress(1.0, desc="生成完成")
        except Exception as e:
            logger.warning(f"更新进度条失败: {str(e)}")
            
        return output_file
    finally:
        # 清理临时文件
        if 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)
            logger.info(f"删除临时图片: {image_path}")
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"删除临时视频: {video_path}")

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# HunyuanPortrait 人像动画生成")
    gr.Markdown("上传一张图片和一个视频，生成人像动画。")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="上传图片")
            video_input = gr.Video(label="上传视频", interactive=True)
        
        with gr.Column():
            gr.Markdown("### 参考文件")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 参考图片")
                    image_files = gr.Gallery(
                        label="参考图片列表",
                        show_label=True,
                        columns=2,
                        rows=2,
                        height=200,
                        object_fit="contain"
                    )
                with gr.Column():
                    gr.Markdown("#### 参考视频")
                    video_files = gr.Dropdown(
                        label="参考视频列表",
                        choices=[],  # 初始化为空
                        value=None,  # 初始化无选中项
                        interactive=True
                    )
    
    generate_btn = gr.Button("生成动画")
    video_output = gr.Video(label="生成的动画")
    
    # 加载参考文件列表
    def load_reference_files():
        image_files, video_names, _ = get_reference_files()
        return gr.update(value=image_files), gr.update(choices=video_names, value=None)
    
    # 选择视频后自动预览
    def load_reference_video(video_name):
        _, _, video_map = get_reference_files()
        video_path = video_map.get(video_name)
        return video_path
    
    # 事件绑定
    video_files.change(load_reference_video, video_files, video_input)
    demo.load(load_reference_files, None, [image_files, video_files])
    
    # 选择图片后自动填充到上传框
    def load_reference_image(evt: gr.SelectData):
        image_files, _, _ = get_reference_files()
        if evt.index is not None and 0 <= evt.index < len(image_files):
            return image_files[evt.index][0]
        return None

    image_files.select(load_reference_image, None, image_input)
    
    generate_btn.click(
        fn=generate_video,
        inputs=[image_input, video_input],
        outputs=video_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866, share=True) 