from typing import Any, Dict
from functools import lru_cache
import threading
import cv2
import numpy
import onnxruntime
from tqdm import tqdm

import facefusion.globals
from facefusion import wording
from facefusion.typing import Frame, ModelValue, Fps
from facefusion.execution_helper import apply_execution_provider_options
from facefusion.vision import get_video_frame, count_video_frame_total, read_image, detect_video_fps
from facefusion.filesystem import resolve_relative_path
from facefusion.download import conditional_download

CONTENT_ANALYSER = None
THREAD_LOCK : threading.Lock = threading.Lock()
MODELS : Dict[str, ModelValue] =\
{
	'open_nsfw':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/open_nsfw.onnx',
		'path': resolve_relative_path('../.assets/models/open_nsfw.onnx')
	}
}
PROBABILITY_LIMIT = 1
RATE_LIMIT = 5
STREAM_COUNTER = 0


def get_content_analyser() -> Any:
	global CONTENT_ANALYSER
	return CONTENT_ANALYSER


def clear_content_analyser() -> None:
	global CONTENT_ANALYSER

	CONTENT_ANALYSER = None


def pre_check() -> bool:
	return True


def analyse_stream(frame : Frame, video_fps : Fps) -> bool:
	global STREAM_COUNTER
	return False


def prepare_frame(frame : Frame) -> Frame:
	frame = cv2.resize(frame, (224, 224)).astype(numpy.float32)
	frame -= numpy.array([ 104, 117, 123 ]).astype(numpy.float32)
	frame = numpy.expand_dims(frame, axis = 0)
	return frame


def analyse_frame(frame : Frame) -> bool:
	return False


@lru_cache(maxsize = None)
def analyse_image(image_path : str) -> bool:
	return False


@lru_cache(maxsize = None)
def analyse_video(video_path : str, start_frame : int, end_frame : int) -> bool:
	return False