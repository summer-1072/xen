import ffmpeg


def video2img(video_path, img_dir):
    (
        ffmpeg
        .input(video_path)
        .output(img_dir, qscale_v=1)
        .run()
    )


def img2video(img_dir, video_path, fps=24):
    (
        ffmpeg
        .input(img_dir, pattern_type='glob', framerate=fps)
        .output(video_path, vcodec='libx264', crf=1)
        .run()
    )
