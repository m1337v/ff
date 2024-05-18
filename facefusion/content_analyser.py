from typing import Any
from functools import lru_cache
from time import sleep
import cv2
import numpy
import onnxruntime
from tqdm import tqdm

import facefusion.globals
from facefusion import process_manager, wording
from facefusion.thread_helper import thread_lock, conditional_thread_semaphore
from facefusion.typing import VisionFrame, ModelSet, Fps
from facefusion.execution import apply_execution_provider_options
from facefusion.vision import get_video_frame, count_video_frame_total, read_image, detect_video_fps
from facefusion.filesystem import resolve_relative_path, is_file
from facefusion.download import conditional_download

CONTENT_ANALYSER = None
MODELS : ModelSet = {}
PROBABILITY_LIMIT = 1
RATE_LIMIT = 10
STREAM_COUNTER = 0


def get_content_analyser() -> Any:
	global CONTENT_ANALYSER
	return None


def clear_content_analyser() -> None:
	global CONTENT_ANALYSER
	CONTENT_ANALYSER = None


def pre_check() -> bool:
	return True


def analyse_stream(vision_frame : VisionFrame, video_fps : Fps) -> bool:
	return False


def analyse_frame(vision_frame : VisionFrame) -> bool:
	return False


def prepare_frame(vision_frame : VisionFrame) -> VisionFrame:
	vision_frame = cv2.resize(vision_frame, (224, 224)).astype(numpy.float32)
	vision_frame -= numpy.array([ 104, 117, 123 ]).astype(numpy.float32)
	vision_frame = numpy.expand_dims(vision_frame, axis = 0)
	return vision_frame


@lru_cache(maxsize = None)
def analyse_image(image_path : str) -> bool:
	return False


@lru_cache(maxsize = None)
def analyse_video(video_path : str, start_frame : int, end_frame : int) -> bool:
	return False
