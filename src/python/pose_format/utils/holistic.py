import numpy as np
from tqdm import tqdm

from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from .openpose import hand_colors, load_frames_directory_dict

try:
    import mediapipe as mp
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES
except ImportError:
    raise ImportError("Please install mediapipe with: pip install mediapipe")

mp_holistic = mp.solutions.holistic

try:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    from mediapipe import Image, ImageFormat
    FACE_LANDMARKER_AVAILABLE = True
except ImportError:
    FACE_LANDMARKER_AVAILABLE = False
    mp_tasks = None
    vision = None
    Image = None
    ImageFormat = None
    Image = None
    ImageFormat = None

FACE_BLEND_SHAPES_NUM = 52
FACE_BLEND_SHAPES_NAMES = [
    '_neutral', '_eyeBlinkLeft', '_eyeLookDownLeft', '_eyeLookInLeft', '_eyeLookOutLeft',
    '_eyeLookUpLeft', '_eyeSquintLeft', '_eyeWideLeft', '_eyeBlinkRight', '_eyeLookDownRight',
    '_eyeLookInRight', '_eyeLookOutRight', '_eyeLookUpRight', '_eyeSquintRight', '_eyeWideRight',
    '_jawForward', '_jawLeft', '_jawOpen', '_jawRight', '_mouthClose', '_mouthFunnel',
    '_mouthPucker', '_mouthLeft', '_mouthRight', '_mouthRollLower', '_mouthRollUpper',
    '_mouthShrugLower', '_mouthShrugUpper', '_mouthPressLeft', '_mouthPressRight',
    '_mouthLowerDownLeft', '_mouthLowerDownRight', '_mouthUpperUpLeft', '_mouthUpperUpRight',
    '_browDownLeft', '_browDownRight', '_browInnerUp', '_browOuterUpLeft', '_browOuterUpRight',
    '_cheekPuff', '_cheekSquintLeft', '_cheekSquintRight', '_noseSneerLeft', '_noseSneerRight',
    '_tongueOut'
]

FACEMESH_CONTOURS_POINTS = [
    str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))
]

BODY_POINTS = mp_holistic.PoseLandmark._member_names_
BODY_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.POSE_CONNECTIONS]

HAND_POINTS = mp_holistic.HandLandmark._member_names_
HAND_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.HAND_CONNECTIONS]

FACE_POINTS_NUM = lambda additional_points=0: additional_points + 468
FACE_POINTS_NUM.__doc__ = """
Gets total number of face points and additional points.

Parameters
----------
additional_points : int, optional
    number of additional points to be added. The defaults is 0.

Returns
-------
int
    total number of face points.
"""
FACE_POINTS = lambda additional_points=0: [str(i) for i in range(FACE_POINTS_NUM(additional_points))]
FACE_POINTS.__doc__ = """
Makes a list of string representations of face points indexes up to total face points number

Parameters
----------
additional_points : int, optional
    number of additional points to be considered. Defaults to 0

Returns
-------
list[str]
    List of strings of face point indices.
"""

FACE_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.FACEMESH_TESSELATION]
FACE_IRISES = [(int(a), int(b)) for a, b in FACEMESH_IRISES]

FLIPPED_BODY_POINTS = [
    'NOSE',
    'RIGHT_EYE_INNER',
    'RIGHT_EYE',
    'RIGHT_EYE_OUTER',
    'LEFT_EYE_INNER',
    'LEFT_EYE',
    'LEFT_EYE_OUTER',
    'RIGHT_EAR',
    'LEFT_EAR',
    'MOUTH_RIGHT',
    'MOUTH_LEFT',
    'RIGHT_SHOULDER',
    'LEFT_SHOULDER',
    'RIGHT_ELBOW',
    'LEFT_ELBOW',
    'RIGHT_WRIST',
    'LEFT_WRIST',
    'RIGHT_PINKY',
    'LEFT_PINKY',
    'RIGHT_INDEX',
    'LEFT_INDEX',
    'RIGHT_THUMB',
    'LEFT_THUMB',
    'RIGHT_HIP',
    'LEFT_HIP',
    'RIGHT_KNEE',
    'LEFT_KNEE',
    'RIGHT_ANKLE',
    'LEFT_ANKLE',
    'RIGHT_HEEL',
    'LEFT_HEEL',
    'RIGHT_FOOT_INDEX',
    'LEFT_FOOT_INDEX',
]


def component_points(component, width: int, height: int, num: int, pose_world_landmarks=False):
    """
    Gets component points

    Parameters
    ----------
    component : object
        Component with landmarks
    width : int
        Width
    height : int
        Height
    num : int
        number of landmarks

    Returns
    -------
    tuple of np.array
        coordinates and confidence for each landmark
    """
    if component is not None:
        lm = component.landmark
        if pose_world_landmarks:
            return np.array([[p.x, p.y, p.z] for p in lm]), np.ones(num)
        else:
            return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.ones(num)       

    return np.zeros((num, 3)), np.zeros(num)


def body_points(component, width: int, height: int, num: int, pose_world_landmarks=False):
    """
    gets body points

    Parameters
    ----------
    component : object
        component containing landmarks
    width : int
        width
    height : int
        Height
    num : int
        number of landmarks

    Returns
    -------
    tuple of np.array
        coordinates and visibility for each landmark.
    """
    if component is not None:
        lm = component.landmark
        if pose_world_landmarks:
            return np.array([[p.x, p.y, p.z] for p in lm]), np.array([p.visibility for p in lm])
        else:
            return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.array([p.visibility for p in lm])            

    return np.zeros((num, 3)), np.zeros(num)


def blend_shapes_points(blend_shapes, num: int):
    """
    Gets face blend shapes values

    Parameters
    ----------
    blend_shapes : list
        List of blend shape objects from MediaPipe
    num : int
        Number of blend shapes

    Returns
    -------
    tuple of np.array
        Blend shape values (as 3D points with value in z) and confidence for each blend shape
    """
    if blend_shapes is not None and len(blend_shapes) > 0:
        # Extract blend shape values
        # Create a mapping from category name to index for faster lookup
        name_to_idx = {name: idx for idx, name in enumerate(FACE_BLEND_SHAPES_NAMES)}
        values = np.zeros(num)
        
        for bs in blend_shapes:
            # Get the category name and find its index
            category_name = bs.category_name if hasattr(bs, 'category_name') else str(bs)
            if category_name in name_to_idx:
                idx = name_to_idx[category_name]
                score = bs.score if hasattr(bs, 'score') else float(bs)
                values[idx] = score
        
        # Store as 3D points: (0, 0, value) format to match XYZC format
        # We use z coordinate to store the blend shape value
        blend_data = np.array([[0.0, 0.0, val] for val in values])
        # Confidence is the blend shape value itself (normalized)
        blend_confidence = values
        return blend_data, blend_confidence

    return np.zeros((num, 3)), np.zeros(num)


def process_holistic(frames: list,
                     fps: float,
                     w: int,
                     h: int,
                     kinect=None,
                     progress=False,
                     additional_face_points=0,
                     additional_holistic_config={},
                     pose_world_landmarks=False,
                     include_face_blend_shapes=False,
                     face_landmarker_model_path=None) -> NumPyPoseBody:
    """
    process frames using holistic model from mediapipe

    Parameters
    ----------
    frames : list
        List of frames to be processed
    fps : float
        Frames per second
    w : int
        Frame width
    h : int
        Frame height.
    kinect : object, optional
        Kinect depth data.
    progress : bool, optional
        If True, show the progress bar.
    additional_face_points : int, optional
        Additional face landmarks (points)
    additional_holistic_config : dict, optional
        Additional configurations for holistic model
    include_face_blend_shapes : bool, optional
        If True, extract face blend shapes using Face Landmarker API
    face_landmarker_model_path : str, optional
        Path to face_landmarker.task model file. Required if include_face_blend_shapes is True.

    Returns
    -------
    NumPyPoseBody
        Processed pose data
    """
    if 'static_image_mode' not in additional_holistic_config:
        additional_holistic_config['static_image_mode'] = False
    holistic = mp_holistic.Holistic(**additional_holistic_config)

    # Initialize Face Landmarker if blend shapes are requested
    face_landmarker = None
    if include_face_blend_shapes:
        if not FACE_LANDMARKER_AVAILABLE:
            raise ImportError("Face Landmarker API not available. Please install mediapipe with: pip install mediapipe")
        if face_landmarker_model_path is None:
            raise ValueError("face_landmarker_model_path is required when include_face_blend_shapes=True")
        
        base_options = mp_tasks.BaseOptions(model_asset_path=face_landmarker_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(options)

    try:
        datas = []
        confs = []

        for i, frame in enumerate(tqdm(frames, disable=not progress)):
            results = holistic.process(frame)

            body_data, body_confidence = body_points(results.pose_landmarks, w, h, 33, pose_world_landmarks)
            face_data, face_confidence = component_points(results.face_landmarks, w, h,
                                                          FACE_POINTS_NUM(additional_face_points), pose_world_landmarks)
            lh_data, lh_confidence = component_points(results.left_hand_landmarks, w, h, 21, pose_world_landmarks)
            rh_data, rh_confidence = component_points(results.right_hand_landmarks, w, h, 21, pose_world_landmarks)
            body_world_data, body_world_confidence = body_points(results.pose_world_landmarks, w, h, 33, pose_world_landmarks)

            # Extract blend shapes if requested
            blend_shapes_data = None
            blend_shapes_confidence = None
            if include_face_blend_shapes and face_landmarker is not None:
                # Convert frame to MediaPipe Image format
                # Ensure frame is RGB (numpy array)
                import cv2
                if len(frame.shape) == 3:
                    # Convert BGR to RGB if needed (OpenCV uses BGR by default)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame
                
                # Create MediaPipe Image (Image is from main mediapipe module, not vision)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
                
                # Process with Face Landmarker
                # Convert frame index to timestamp in milliseconds
                timestamp_ms = int(i * 1000 / fps) if fps > 0 else i
                face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
                    blend_shapes_data, blend_shapes_confidence = blend_shapes_points(
                        face_result.face_blendshapes[0], FACE_BLEND_SHAPES_NUM
                    )
                else:
                    blend_shapes_data, blend_shapes_confidence = blend_shapes_points(None, FACE_BLEND_SHAPES_NUM)

            # Concatenate data
            if include_face_blend_shapes and blend_shapes_data is not None:
                data = np.concatenate([body_data, face_data, lh_data, rh_data, body_world_data, blend_shapes_data])
                conf = np.concatenate([body_confidence, face_confidence, lh_confidence, rh_confidence, body_world_confidence, blend_shapes_confidence])
            else:
                data = np.concatenate([body_data, face_data, lh_data, rh_data, body_world_data])
                conf = np.concatenate([body_confidence, face_confidence, lh_confidence, rh_confidence, body_world_confidence])

            if kinect is not None:
                kinect_depth = []
                for x, y, z in np.array(data, dtype="int32"):
                    if 0 < x < w and 0 < y < h:
                        kinect_depth.append(kinect[i, y, x, 0])
                    else:
                        kinect_depth.append(0)

                kinect_vec = np.expand_dims(np.array(kinect_depth), axis=-1)
                data = np.concatenate([data, kinect_vec], axis=-1)

            datas.append(data)
            confs.append(conf)

        pose_body_data = np.expand_dims(np.stack(datas), axis=1)
        pose_body_conf = np.expand_dims(np.stack(confs), axis=1)

        return NumPyPoseBody(data=pose_body_data, confidence=pose_body_conf, fps=fps)
    finally:
        holistic.close()
        if face_landmarker is not None:
            face_landmarker.close()


def holistic_hand_component(name, pf="XYZC") -> PoseHeaderComponent:
    """
    Creates holistic hand component

    Parameters
    ----------
    name : str
        Component name
    pf : str, optional
        Point format

    Returns
    -------
    PoseHeaderComponent
        Hand component
    """
    return PoseHeaderComponent(name=name, points=HAND_POINTS, limbs=HAND_LIMBS, colors=hand_colors, point_format=pf)


def holistic_components(pf="XYZC", additional_face_points=0, include_face_blend_shapes=False):
    """
    Creates list of holistic components

    Parameters
    ----------
    pf : str, optional
        Point format
    additional_face_points : int, optional
        Additional face points/landmarks
    include_face_blend_shapes : bool, optional
        If True, include FACE_BLEND_SHAPES component

    Returns
    -------
    list of PoseHeaderComponent
        List of holistic components.
    """
    face_limbs = list(FACE_LIMBS)
    if additional_face_points > 0:
        face_limbs += FACE_IRISES

    components = [
        PoseHeaderComponent(name="POSE_LANDMARKS",
                            points=BODY_POINTS,
                            limbs=BODY_LIMBS,
                            colors=[(255, 0, 0)],
                            point_format=pf),
        PoseHeaderComponent(name="FACE_LANDMARKS",
                            points=FACE_POINTS(additional_face_points),
                            limbs=face_limbs,
                            colors=[(128, 0, 0)],
                            point_format=pf),
        holistic_hand_component("LEFT_HAND_LANDMARKS", pf),
        holistic_hand_component("RIGHT_HAND_LANDMARKS", pf),
        PoseHeaderComponent(name="POSE_WORLD_LANDMARKS",
                            points=BODY_POINTS,
                            limbs=BODY_LIMBS,
                            colors=[(255, 0, 0)],
                            point_format=pf),
    ]
    
    # Add face blend shapes component if requested
    if include_face_blend_shapes:
        components.append(
            PoseHeaderComponent(name="FACE_BLEND_SHAPES",
                                points=FACE_BLEND_SHAPES_NAMES,
                                limbs=[],  # Blend shapes don't have limbs
                                colors=[(200, 100, 0)],  # Orange color for blend shapes
                                point_format=pf)
        )
    
    return components


def load_holistic(frames: list,
                  fps: float = 24,
                  width=1000,
                  height=1000,
                  depth=0,
                  kinect=None,
                  progress=False,
                  additional_holistic_config={},
                  pose_world_landmarks=False,
                  include_face_blend_shapes=False,
                  face_landmarker_model_path=None) -> Pose:
    """
    Loads holistic pose data

    Parameters
    ----------
    frames : list
        List of frames.
    fps : float, optional
        Frames per second.
    width : int, optional
        Frame width.
    height : int, optional
        Frame height.
    depth : int, optional
        Depth data.
    kinect : object, optional
        Kinect depth data.
    progress : bool, optional
        If True, show the progress bar.
    additional_holistic_config : dict, optional
        Additional configurations for the holistic model.
    include_face_blend_shapes : bool, optional
        If True, extract and include face blend shapes in the pose data.
    face_landmarker_model_path : str, optional
        Path to face_landmarker.task model file. Required if include_face_blend_shapes is True.
        Can be downloaded from: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

    Returns
    -------
    Pose
        Loaded pose data with header and body 
    """
    pf = "XYZC" if kinect is None else "XYZKC"

    dimensions = PoseHeaderDimensions(width=width, height=height, depth=depth)

    refine_face_landmarks = 'refine_face_landmarks' in additional_holistic_config and additional_holistic_config[
        'refine_face_landmarks']
    additional_face_points = 10 if refine_face_landmarks else 0
    header: PoseHeader = PoseHeader(version=0.2,
                                    dimensions=dimensions,
                                    components=holistic_components(pf, additional_face_points, include_face_blend_shapes))
    body: NumPyPoseBody = process_holistic(frames, fps, width, height, kinect, progress, additional_face_points,
                                           additional_holistic_config, pose_world_landmarks,
                                           include_face_blend_shapes, face_landmarker_model_path)

    return Pose(header, body)


def formatted_holistic_pose(width: int, height: int, additional_face_points: int = 0, include_face_blend_shapes: bool = False):
    """
    Formatted holistic pose

    Parameters
    ----------
    width : int
        Pose width.
    height : int
        Pose height.
    additional_face_points : int, optional
        Additional face points/landmarks.
    include_face_blend_shapes : bool, optional
        If True, include FACE_BLEND_SHAPES in the returned components.

    Returns
    -------
    object
        Formatted pose components
    """
    dimensions = PoseHeaderDimensions(width=width, height=height, depth=1000)
    header = PoseHeader(version=0.2,
                        dimensions=dimensions,
                        components=holistic_components("XYZC", additional_face_points, include_face_blend_shapes))
    body = NumPyPoseBody(
        fps=0,  # to be overridden later
        data=np.zeros(shape=(1, 1, header.total_points(), 3)),
        confidence=np.zeros(shape=(1, 1, header.total_points())))
    pose = Pose(header, body)
    
    # Build component list dynamically to include FACE_BLEND_SHAPES if present
    components_to_get = ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]
    if include_face_blend_shapes:
        components_to_get.append("FACE_BLEND_SHAPES")
    
    return pose.get_components(components_to_get,
                               {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS})


def load_mediapipe_directory(directory: str, fps: int, width: int, height: int, num_face_points: int = 128) -> Pose:
    """
    Load pose data from a directory of MediaPipe

    Parameters
    ----------
    directory : str
        Directory path.
    fps : float
        Frames per second.
    width : int
        Frame width.
    height : int
        Frame height.
    num_face_points : int, optional
        Number of face landmarks. Ideally, we don't want to hard code the 128 for the face, since face points can be 128 (reduced with refinement) or 118 (reduced without refinement) or 478 (full with refinement) or 468 (full without refinement)

    Returns
    -------
    Pose
        Loaded pose data
    """

    frames = load_frames_directory_dict(directory=directory, pattern="(?:^|\D)?(\d+).*?.json")

    if len(frames) > 0:
        first_frame = frames[0]
        num_pose_points = first_frame["pose_landmarks"]["num_landmarks"]
        num_left_hand_points = first_frame["left_hand_landmarks"]["num_landmarks"]
        num_right_hand_points = first_frame["right_hand_landmarks"]["num_landmarks"]
        additional_face_points = 10 if (num_face_points == 478 or num_face_points == 128) else 0
    else:
        raise ValueError("No frames found in directory: {}".format(directory))

    def load_mediapipe_frame(frame):
        """
        Get landmarks data of face landmarks, pose landmarks, and left & right hand landmarks (body_data, face_data, lh_data, rh_data) and confidence values from a given frame.

    Parameters
    ----------
    frame : dict
        Dictionary containing face, pose, left hand, and right hand landmark data.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two arrays:
        The first array is the landmarks data including x, y, z coordinates. 
        The second array is the confidence scores for each landmark.
         """

        def load_landmarks(name, num_points: int):
            points = [[float(p) for p in r.split(",")] for r in frame[name]["landmarks"]]
            points = [(ps + [1.0])[:4] for ps in points]  # Add visibility to all points
            if len(points) == 0:
                points = [[0, 0, 0, 0] for _ in range(num_points)]
            return np.array([[x, y, z] for x, y, z, c in points]), np.array([c for x, y, z, c in points])

        face_data, face_confidence = load_landmarks("face_landmarks", num_face_points)
        body_data, body_confidence = load_landmarks("pose_landmarks", num_pose_points)
        lh_data, lh_confidence = load_landmarks("left_hand_landmarks", num_left_hand_points)
        rh_data, rh_confidence = load_landmarks("right_hand_landmarks", num_right_hand_points)
        data = np.concatenate([body_data, face_data, lh_data, rh_data])
        conf = np.concatenate([body_confidence, face_confidence, lh_confidence, rh_confidence])
        return data, conf

    def load_mediapipe_frames() -> NumPyPoseBody:
        """
        From a list of frames, load  pose data and confidance into a NumPyPoseBody
        
        Processes each frame from `frames` to extract the data and confidence values
        for pose landmarks, face landmarks, and left & right hand landmarks.

        Returns
        -------
        NumPyPoseBody
            PoseBody object with data and confidence for each frame.
        """
        max_frames = int(max(frames.keys())) + 1
        pose_body_data = np.zeros(shape=(max_frames, 1, num_left_hand_points + num_right_hand_points + num_pose_points +
                                         num_face_points, 3),
                                  dtype=float)
        pose_body_conf = np.zeros(shape=(max_frames, 1, num_left_hand_points + num_right_hand_points + num_pose_points +
                                         num_face_points),
                                  dtype=float)
        for frame_id, frame in frames.items():
            data, conf = load_mediapipe_frame(frame)
            pose_body_data[frame_id][0] = data
            pose_body_conf[frame_id][0] = conf
        return NumPyPoseBody(data=pose_body_data, confidence=pose_body_conf, fps=fps)

    pose = formatted_holistic_pose(width=width, height=height, additional_face_points=additional_face_points)

    pose.body = load_mediapipe_frames()

    return pose
