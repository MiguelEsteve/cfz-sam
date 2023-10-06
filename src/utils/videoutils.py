import math
import os
import time
import datetime
import numpy as np
import cv2 as cv2
from typing import Callable
from blinker import signal
import ffmpeg
from PIL import Image, ImageDraw


from configs.log_conf import getLogger

LOGGER = getLogger(__name__)


class ImageUtils:
    @staticmethod
    def read_image(image_file: str, use: str = 'cv2'):
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"{os.path.basename(image_file)} ")
        if use == "PIL":
            img = Image.open(image_file)
            return np.asarray(img)
        if use == 'cv2':
            return cv2.imread(image_file)
        else:
            raise Exception(f"invalid use: {use}")


class VideoUtils:
    _STARTED_SIGNAL = signal("started")
    _PROGRES_SIGNAL = signal("progress")
    _FINISHED_SIGNAL = signal("finished")

    @staticmethod
    def get_frames(video_fn, width=48, height=27, add_frame_number=False, factor=3, start=0, end=np.inf,
                   num_frames=np.inf,
                   backend="opencv"):
        if backend == "opencv":
            return VideoUtils.get_frames_opencv(video_fn=video_fn,
                                                width=width,
                                                height=height,
                                                add_frame_number=add_frame_number,
                                                start=start,
                                                end=end,
                                                num_frames=num_frames)
        elif backend == "ffmpeg":
            return VideoUtils.get_frames_ffmpeg(video_fn=video_fn,
                                                width=width,
                                                height=height,
                                                add_frame_number=add_frame_number,
                                                factor=factor,
                                                start=start,
                                                end=end,
                                                num_frames=num_frames)

    @staticmethod
    def get_frames_ffmpeg(video_fn, width=48, height=27, add_frame_number=False, factor=3, start=0, end=np.inf,
                          num_frames=np.inf):
        """
        Provides all the frames contained in a mp5 video file.
        Parameters
        --------------------------
            video_fn: path to mp4 video file
            width: width in pixels of each frame
            height: height in pixels of each frame
            add_frame_number: if adding the frame number starting in 0
            start: start frame to be retrieved
            end: end frame to be retrieved
            num_frames: number of frames we want to retrieve,
        Returns
        ---------------------------
            numpy array of shape [n_frames, height, width, channels]
        """

        t1 = time.time()
        total_frames = VideoUtils.get_frame_count(video_fn)

        w, h = VideoUtils.get_frame_resolution(video_fn)
        fontsize = min(w, h) // factor

        if total_frames == -1:
            return -1
        assert start < total_frames, \
            f"not enough frames in video, total: {total_frames} vs start: {start}"

        if end != np.inf:
            num_frames = end + 1
        elif num_frames == np.inf:
            num_frames = total_frames

        try:
            if add_frame_number:
                video_stream, err = (
                    ffmpeg
                    .input(video_fn)
                    .drawtext(box=True, fontfile="c:\windows\Fonts\Calibri.ttf", boxcolor="black", borderw=10,
                              text="%{n}", fontcolor="white", fontsize=str(fontsize), escape_text=False)
                    .output('pipe:',
                            format='rawvideo',
                            pix_fmt='rgb24',
                            s='{}x{}'.format(width, height),
                            frames=num_frames)
                    .run_on_image(capture_stdout=True, capture_stderr=True)
                )
            else:
                video_stream, err = (
                    ffmpeg
                    .input(video_fn)
                    .output('pipe:',
                            format='rawvideo',
                            pix_fmt='rgb24',
                            s='{}x{}'.format(width, height),
                            frames=num_frames)
                    .run_on_image(capture_stdout=True, capture_stderr=True)
                )
            video = np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])

            if "bbc_01" in video_fn:
                video = video[1:]

            if end == np.inf and num_frames != np.inf:
                end = num_frames + start  # TODO change to end = num_frames + start
            LOGGER.info(f"frames loaded - slicing - start: {start}, end: {end}")
            LOGGER.info(f"frames loaded - video shape before slicing: {video.shape}")

            elapsed = datetime.timedelta(seconds=math.ceil(time.time() - t1))
            LOGGER.debug(f"{video_fn} loaded to array; elapsed time: {str(elapsed)}")

            return video[start:end]
        except Exception as e:
            LOGGER.error(f"ffmpeg std out: {e.stdout.decode('utf-8')}")
            LOGGER.error(f"ffmpeg error: {e.stderr.decode('utf-8')}")
            LOGGER.error(f"ffmpeg error with {video_fn}")
            return -1

    @staticmethod
    def get_frames_opencv(video_fn, width=48, height=27, add_frame_number=False, start=0, end=np.inf,
                          num_frames=np.inf):
        """
        Provides all the frames contained in a mp5 video file.
        Parameters
        --------------------------
            video_fn: path to mp4 video file
            width: width in pixels of each frame
            height: height in pixels of each frame
            add_frame_number: if adding the frame number starting in 0
            start: start frame to be retrieved
            end: end frame to be retrieved
            num_frames: number of frames we want to retrieve,
        Returns
        ---------------------------
            numpy array of shape [n_frames, height, width, channels]
        """
        t1 = time.time()

        frames = []
        cap = cv2.VideoCapture(video_fn)
        if not cap.isOpened():
            logging.debug("File could not be opened")
            return -1

        total_frames = cap.get(propId=cv2.CAP_PROP_FRAME_COUNT)
        VideoUtils._STARTED_SIGNAL.send(total_frames)

        ret = True
        count = 0
        while ret:
            ret, frame = cap.read()  # frame is a np.array dtype uint8
            if ret:
                if count == 0:
                    VideoUtils._PROGRES_SIGNAL.send(1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (width, height))
                if add_frame_number:
                    frame = cv2.rectangle(frame, (0, 0), (15, 5), (0, 0, 0), thickness=cv2.FILLED)
                    cv2.putText(frame, str(count), (0, 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255))
                frames.append(frame)
                count += 1
                if count != 0:
                    VideoUtils._PROGRES_SIGNAL.send(count)

        frames = np.stack(frames, axis=0)
        if "bbc_01" in video_fn:
            frames = frames[1:]
        VideoUtils._FINISHED_SIGNAL.send(frames.shape[0])

        elapsed = datetime.timedelta(seconds=math.ceil(time.time() - t1))
        LOGGER.debug(f"{video_fn} loaded to array; elapsed time: {str(elapsed)}")

        if end == np.inf and num_frames != np.inf:
            end = num_frames + start
            return frames[start:end]
        elif end != np.inf:
            return frames[start:end]
        else:
            return frames[start:]

    @staticmethod
    def get_frames_simple(video_fn, width=48, height=27, scale: float = None, add_frame_number=False, num_frames=None,
                          read_frames_skipped: int = 25):

        t1 = time.time()
        frames = []
        cap = cv2.VideoCapture(video_fn)
        if not cap.isOpened():
            logging.debug("File could not be opened")
            return -1

        if scale is not None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

        total_frames = cap.get(propId=cv2.CAP_PROP_FRAME_COUNT)
        VideoUtils._STARTED_SIGNAL.send(total_frames)

        FR = cap.get(cv2.CAP_PROP_FPS)

        if read_frames_skipped is not None:
            assert 0 < read_frames_skipped <= FR, f"frame rate is {FR} and value required {read_frames_skipped}"

        skip_factor = FR - read_frames_skipped + 1 if read_frames_skipped is not None else 1
        LOGGER.debug(
            f"Frame rate: {FR}, read frames skipped set to {read_frames_skipped}, calculated skip factor: "
            f"{skip_factor}")

        count = 0
        skip_count = 0
        frames_count = 0
        while True:
            ret, frame = cap.read()  # frame is a np.array dtype uint8
            if not ret:
                break
            if frames_count == 0:
                VideoUtils._PROGRES_SIGNAL.send(1)

            if skip_count % skip_factor == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if add_frame_number:
                    if width is not None and height is not None:
                        frame = cv2.resize(frame, (width, height))
                    frame = DisplayUtils.draw_text_on_rect(frame, str(count), position=(50, 50))
                frames.append(frame)
                skip_count = 0
                frames_count += 1
                if frames_count != 0:
                    VideoUtils._PROGRES_SIGNAL.send(frames_count)
                if num_frames is not None and frames_count >= num_frames:
                    break

            skip_count += 1
            count += 1

        frames = np.stack(frames, axis=0)
        if "bbc_01" in video_fn:
            frames = frames[1:]

        VideoUtils._FINISHED_SIGNAL.send(frames.shape[0])

        elapsed = datetime.timedelta(seconds=math.ceil(time.time() - t1))
        LOGGER.debug(f"{video_fn} loaded to array; elapsed time: {str(elapsed)}")
        return frames, {'total_frames': total_frames, 'frame_rate': FR, 'read_fremes_skipped': read_frames_skipped,
                        'duration': math.ceil(total_frames / FR)}

    @staticmethod
    def get_frame_count(fn):
        """
        Provides the number of frames in a mp4 video file.
        :param fn: path to mp4 video file
        :return: number of video frames
        """
        try:
            probe = ffmpeg.probe(fn)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            num_frames = int(video_stream["nb_frames"])
            return num_frames
        except Exception:
            LOGGER.debug(f"ffmpeg probe error with {fn}")
            return -1

    @staticmethod
    def get_frame_resolution(fn):
        """
        Provides the frame resolution in a mp4 video file
        Parameters
        ----------
        fn: path to mp4 vide file

        Returns
        -------
        w, y: horizontal and vertical resolution
        """
        try:
            probe = ffmpeg.probe(fn)
            print(probe)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            return video_stream['width'], video_stream['height']
        except Exception:
            LOGGER.debug(f"ffmpeg probe error with {fn}")
            return -1

    @staticmethod
    def get_video_data(fn):
        data = {}
        try:
            probe = ffmpeg.probe(fn)
            data['num_video_streams'] = len([stream for stream in probe['streams'] if stream['codec_type'] == 'video'])
            data['num_audio_streams'] = len([stream for stream in probe['streams'] if stream['codec_type'] == 'audio'])
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            data['video_codec'] = video_stream['codec_long_name']
            data['video_profile'] = video_stream['profile']
            data['audio_codec'] = audio_stream['codec_long_name']
            data['audio_profile'] = audio_stream['profile']
            data['avg_frame_rate'] = video_stream['avg_frame_rate']
            data['duration'] = video_stream['duration']
            data['num_frames'] = video_stream['nb_frames']
            data['width'] = video_stream['width']
            data['height'] = video_stream['height']
            data['ar'] = video_stream['display_aspect_ratio']
            data['pix_fmt'] = video_stream['pix_fmt']
            data['bitsperrawsample'] = video_stream['bits_per_raw_sample']

            s = ""
            for k, v in data.items():
                s += f"{k}: {v}\n"

            return data, s

        except Exception:
            LOGGER.debug(f"ffmpeg probe error with {fn}")
            return -1

    @staticmethod
    def convert(source: str, target: str):
        def find_files_with_extensions(root_dir: str, extension: str):
            matches = []
            if not extension.startswith("."):
                extension = "." + extension
            for dirpath, dirnames, filenames in os.walk(root_dir):
                if "spain_laliga" in dirpath:
                    for filename in filenames:
                        if "720" in filename:
                            if filename.endswith(extension):
                                matches.append(os.path.join(dirpath, filename))
            return matches

        mkvs = find_files_with_extensions(root_dir=source, extension=".mkv")

        for i, mkv in enumerate(mkvs):
            output_file = os.path.join(target, os.path.basename(mkv).replace(".mkv", f"_{i}.mp4"))

        stream = ffmpeg.input(mkv, ss=10, t=100)
        stream = ffmpeg.output(stream, output_file, vcodec='libx264', acodec='aac')
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)

    @staticmethod
    def loop_video(video_fn: str, add_frame_number=False, read_frames_skipped: int = None, num_frames: int = None,
                   display: bool = False, save_file: str = None, func: Callable = None):

        t1 = time.time()

        cap = cv2.VideoCapture(video_fn)
        if not cap.isOpened():
            LOGGER.error(f"Error, could not open file {video_fn}")
            exit()
        FR = cap.get(cv2.CAP_PROP_FPS)

        if read_frames_skipped is not None:
            assert 0 < read_frames_skipped < FR, f"frame rate is {FR} and value required {read_frames_skipped}"

        skip_factor = FR - read_frames_skipped + 1 if read_frames_skipped is not None else 1
        LOGGER.debug(
            f"Frame rate: {FR}, read frame rate set to {read_frames_skipped}, calculated skip factor: {skip_factor}")

        skip_count = 0
        frames_count = 0
        count_num_frames = 0

        processed_frames = []

        while True:
            ret, frame = cap.read()  # Read a frame from the video
            if not ret:
                break  # Break the loop fo we've reached the end of the video

            if skip_count % skip_factor == 0:
                if add_frame_number:
                    frame = DisplayUtils.draw_text_on_rect(frame, str(frames_count), position=(2, 10), font_scale=0.6,
                                                           padding=4)

                frame = func(frame, frames_count) if func is not None else frame
                processed_frames.append(frame)
                if display:
                    cv2.imshow('Video frame', frame)
                skip_count = 0
                count_num_frames += 1
            frames_count += 1
            skip_count += 1
            if cv2.waitKey(int(FR)) & 0xFF == ord('q'):
                break
            if num_frames is not None and count_num_frames >= num_frames:
                break

        cap.release()
        cv2.destroyAllWindows()

        all_frames = np.stack(processed_frames)

        elapsed = datetime.timedelta(seconds=math.ceil(time.time() - t1))

        LOGGER.debug(f'Resnet50 inference on "{video_fn}" took {elapsed}')

        if save_file:
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
            DisplayUtils.convert_frames_to_mosaic(frames=all_frames, save_file=save_file)
        return all_frames

    @staticmethod
    def extract_frames_from_video(video_fn, output_path, add_frame_number=False, read_frames_skipped: int = None, num_frames: int = None):
        def func(frame: np.ndarray, idx:int):
            file_path = os.path.join(output_path, os.path.basename(video_fn)[:-4] + f'_frame_{idx}.jpg')
            cv2.imwrite(file_path, frame)
            return frame
        VideoUtils.loop_video(video_fn=video_fn, add_frame_number=add_frame_number, read_frames_skipped=read_frames_skipped, num_frames=num_frames, func=func)


class DisplayUtils:
    def __init__(self):
        pass

    @staticmethod
    def draw_text_on_rect(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0,
                          text_color=(255, 255, 255), rect_color=(0, 0, 0), thickness=2, padding=10):
        """
        Draw a rectangle and overlay text on it.

        Args:
        - image (numpy.ndarray): Image to draw on.
        - text (str): The text string to overlay.
        - position (tuple): The top-left position (x, y) where the rectangle should start.
        - font, font_scale, text_color, thickness: Text properties.
        - rect_color: Color of the rectangle.
        - padding: Padding around the text inside the rectangle.

        Returns:
        - numpy.ndarray: Image with rectangle and text overlay.
        """

        # Get the size of the text
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Define the bottom-right position for the rectangle
        rect_bottom_right = (position[0] + text_width + 2 * padding, position[1] + text_height + 2 * padding)

        # Draw the rectangle
        cv2.rectangle(image, position, rect_bottom_right, rect_color, -1)  # -1 fills the rectangle

        # Calculate the position to start the text such that it's centered in the rectangle
        text_pos = (position[0] + padding, position[1] + text_height + padding)

        # Overlay the text
        cv2.putText(image, text, text_pos, font, font_scale, text_color, thickness, cv2.LINE_AA)

        return image

    @staticmethod
    def display_video(video_fn: str, add_frame_number=False,
                      read_frames_skipped: int = None):

        cap = cv2.VideoCapture(video_fn)
        if not cap.isOpened():
            LOGGER.error(f"Error, could not open file {video_fn}")
            exit()
        FR = cap.get(cv2.CAP_PROP_FPS)

        if read_frames_skipped is not None:
            assert 0 < read_frames_skipped < FR, f"frame rate is {FR} and value required {read_frames_skipped}"

        skip_factor = FR - read_frames_skipped + 1 if read_frames_skipped is not None else 1
        LOGGER.debug(
            f"Frame rate: {FR}, read frame rate set to {read_frames_skipped}, calculated skip factor: {skip_factor}")

        skip_count = 0
        frames_count = 0

        while True:
            ret, frame = cap.read()  # Read a frame from the video
            if not ret:
                break  # Break the loop fo we've reached the end of the video

            if skip_count % skip_factor == 0:
                if add_frame_number:
                    frame = DisplayUtils.draw_text_on_rect(frame, str(frames_count), position=(50, 50))
                cv2.imshow('Video frame', frame)
                skip_count = 0
            frames_count += 1
            skip_count += 1
            if cv2.waitKey(int(FR)) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def convert_frames_to_mosaic(frames: np.ndarray, width=100, regul=True, pad=0, save_file:str = None):
        nf, ih, iw, ic = frames.shape
        t = nf // 10 if nf//10 !=0 else 1
        width = np.minimum(width, t) if regul else width
        if nf % width:
            pad_width = width - nf % width
            frames = np.concatenate([frames, np.ones(shape=(pad_width, ih, iw, ic), dtype=np.uint8) * pad])

        height = frames.shape[0] // width
        frames = frames.reshape([height, width, ih, iw, ic])

        frames = np.split(frames, height)
        frames = np.concatenate(frames, axis=2)[0]
        frames = np.split(frames, width)
        frames = np.concatenate(frames, axis=2)[0]
        if save_file is not None:
            cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frames)
            img.save(save_file)

        return frames

    @staticmethod
    def show_frames_from_array(frames: np.ndarray, widthShow=10):
        t1 = time.time()

        frames = DisplayUtils.convert_frames_to_mosaic(frames, width=widthShow).astype(np.uint8)
        img = Image.fromarray(frames)
        draw = ImageDraw.Draw(img, "RGBA")
        img.show()

        elapsed_time = datetime.timedelta(seconds=math.ceil(time.time() - t1))
        LOGGER.debug(f"Total time to show using PIL is {elapsed_time}")

        return img, draw

    @staticmethod
    def display_mosaic(video_fn: str, add_frame_number=False, width: int = None, height: int = None,
                       scale: float = None, num_frames: int = None,
                       read_frames_skipped: int = None):
        video, _ = VideoUtils.get_frames_simple(video_fn=video_fn,
                                                width=width,
                                                height=height,
                                                scale=scale,
                                                add_frame_number=add_frame_number,
                                                num_frames=num_frames,
                                                read_frames_skipped=read_frames_skipped)
        DisplayUtils.show_frames_from_array(video)

