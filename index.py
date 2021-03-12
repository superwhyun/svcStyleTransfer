import streamlit as st
import adain_style_transfer as ast
from view_util import download_file
from tempfile import NamedTemporaryFile
from PIL import Image
import io
import numpy as np

stn = ast.AdainStyleTransfer()


def viewStyleTransfer():
    st.title("Style Transfer")
    alpha = st.slider("Choose alpha value: ", min_value=0.0, max_value=1.0, value=0.5, step=0.1)


    content_file = st.file_uploader("Choose an image for Content", type=["jpg"])
    style_file = st.file_uploader("Choose an image for Style", type=["jpg"])

    col1, col2, col3 = st.beta_columns(3)

    content_image = None
    style_image = None
    
    with col1:
        st.header("Content")
        if(content_file is not None):
            image = content_file.read()
            st.image(image, caption='your content', use_column_width=True)
            #st.write(content_file.name)
            content_image=image

                #temp_file = NamedTemporaryFile(delete=False)
                #temp_file.write(image)
                #print(temp_file.name)
    with col2:
        st.header("Style")    
        if(style_file is not None):
            image = style_file.read()
            st.image(image, caption='your style', use_column_width=True)
            st.write(style_file.name)
            style_image=image
    with col3:
        st.header("Result")
        if(content_file is not None and style_file is not None):
            output = stn.style_transfer( Image.open(io.BytesIO(content_image)),  Image.open(io.BytesIO(style_image)), alpha)
            image=Image.fromarray(output)
            st.image(image, caption='transferred image', use_column_width=True)
                        

import av
import cv2

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)


def viewWebRTCVideoTransform():    
    st.title("Live Video Transformation using WebRTC")

    """ Video transforms with OpenCV """

    class OpenCVVideoTransformer(VideoTransformerBase):
        type: Literal["noop", "cartoon", "edges", "rotate"]

        def __init__(self) -> None:
            self.type = "noop"

        """ 
        -> : 사용자 정의 타입에 대한 힌트를 주는 것 
        https://www.daleseo.com/python-type-annotations/
        https://www.daleseo.com/python-typing/
        """
        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:

            """ 
            An ndarray is a (usually fixed-size) multidimensional container of items of the same type and size. 
            The number of dimensions and items in an array is defined by its shape , which is a tuple of N non-negative integers 
            that specify the sizes of each dimension.
            """
            img = frame.to_ndarray(format="bgr24")

            if self.type == "noop":
                pass
            elif self.type == "cartoon":
                # prepare color
                """
                동일한 영상에 대해서 이미지의 크기를 50% 크기만큼 연속적으로 줄이거나 늘려 생성된 이미지 집합을 
                이미지 피라미드라고 합니다. OpenCV에서 제공하는 이미지 피라미드 관련 함수는 cv2.pyrDown과 cv2.pryUp 인데, 
                cv2.pyrDown 함수는 입력 이미지를 50% 크기로 줄인 이미지를 생성해 반환하고 cv2.pryUp은 입력 이미지를 
                200% 크기로 확대한 이미지를 생성해 반환합니다. "http://www.gisdeveloper.co.kr/?p=6560"
                """
                img_color = cv2.pyrDown(cv2.pyrDown(img))
                for _ in range(6):
                    img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                # prepare edges
                img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_edges = cv2.adaptiveThreshold(
                    cv2.medianBlur(img_edges, 7),
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    9,
                    2,
                )
                img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

                # combine color and edges
                img = cv2.bitwise_and(img_color, img_edges)
            elif self.type == "edges":
                # perform edge detection
                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
            elif self.type == "rotate":
                # rotate image
                rows, cols, _ = img.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
                img = cv2.warpAffine(img, M, (cols, rows))

            return img

    
    # WebRTC Stream을 생성하고 transform 함수를 등록
    

    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=OpenCVVideoTransformer,
        async_transform=True,
    )

    transform_type = st.radio(
        "Select transform type", ("noop", "cartoon", "edges", "rotate")
    )

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.type = transform_type

    st.markdown(
        "This demo is based on "
        "https://github.com/aiortc/aiortc/blob/2362e6d1f0c730a0f8c387bbea76546775ad2fe8/examples/server/server.py#L34. "  # noqa: E501
        "https://github.com/whitphx/streamlit-webrtc-example/"
        "Many thanks to the project."
    )


from pathlib import Path
import queue
from typing import List, NamedTuple

HERE = Path(__file__).parent

def viewLiveObjectDetect():
    """Object detection demo with MobileNet SSD.
    This model and code are based on
    https://github.com/robmarkcole/object-detection-app
    """
    MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
    MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
    PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
    PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    class MobileNetSSDVideoTransformer(VideoTransformerBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            self._net = cv2.dnn.readNetFromCaffe(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, detections):
            # loop over the detections
            (h, w) = image.shape[:2]
            result: List[Detection] = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    # extract the index of the class label from the `detections`,
                    # then compute the (x, y)-coordinates of the bounding box for
                    # the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    name = CLASSES[idx]
                    result.append(Detection(name=name, prob=float(confidence)))

                    # display the prediction
                    label = f"{name}: {round(confidence * 100, 2)}%"
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        image,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[idx],
                        2,
                    )
            return image, result

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            image = frame.to_ndarray(format="bgr24")

            #test code
            '''
            img = cv2.cvtColor(cv2.Canny(image, 100, 200), cv2.COLOR_GRAY2BGR)            
            return img
            '''
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
            )
            self._net.setInput(blob)
            detections = self._net.forward()
            annotated_image, result = self._annotate_image(image, detections)

            # NOTE: This `transform` method is called in another thread,
            # so it must be thread-safe.
            self.result_queue.put(result)

            return annotated_image
            

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=MobileNetSSDVideoTransformer,
        async_transform=True,
    )

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.confidence_threshold = confidence_threshold

    
    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            
            while True:
                if webrtc_ctx.video_transformer:    
                    result = webrtc_ctx.video_transformer.result_queue.get()
                    labels_placeholder.table(result)
                else:
                    break
    

    st.markdown(
        "This demo uses a model and code from "
        "https://github.com/robmarkcole/object-detection-app. "
        "Many thanks to the project."
    )


from aiortc.contrib.media import MediaPlayer

def viewMediaStreaming():
    """ Media streamings """
    MEDIAFILES = {
        "big_buck_bunny_720p_2mb.mp4": {
            "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_2mb.mp4",  # noqa: E501
            "local_file_path": HERE / "data/big_buck_bunny_720p_2mb.mp4",
            "type": "video",
        },
        "big_buck_bunny_720p_10mb.mp4": {
            "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_10mb.mp4",  # noqa: E501
            "local_file_path": HERE / "data/big_buck_bunny_720p_10mb.mp4",
            "type": "video",
        },
        "file_example_MP3_700KB.mp3": {
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3",  # noqa: E501
            "local_file_path": HERE / "data/file_example_MP3_700KB.mp3",
            "type": "audio",
        },
        "file_example_MP3_5MG.mp3": {
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_5MG.mp3",  # noqa: E501
            "local_file_path": HERE / "data/file_example_MP3_5MG.mp3",
            "type": "audio",
        },
    }
    media_file_label = st.radio(
        "Select a media file to stream", tuple(MEDIAFILES.keys())
    )
    media_file_info = MEDIAFILES[media_file_label]
    download_file(media_file_info["url"], media_file_info["local_file_path"])

    def create_player():
        return MediaPlayer(str(media_file_info["local_file_path"]))

        # NOTE: To stream the video from webcam, use the code below.
        # return MediaPlayer(
        #     "1:none",
        #     format="avfoundation",
        #     options={"framerate": "30", "video_size": "1280x720"},
        # )

    WEBRTC_CLIENT_SETTINGS.update(
        {
            "media_stream_constraints": {
                "video": media_file_info["type"] == "video",
                "audio": media_file_info["type"] == "audio",
            }
        }
    )

    print(media_file_label)
    webrtc_streamer(
        key=f"media-streaming-{media_file_label}",
        mode=WebRtcMode.RECVONLY,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        player_factory=create_player,
    )



def main():
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "StyleTransfer", "LiveVideoTransform", "LiveObjectDetect", "MediaStreaming", "Segmentation"])

    if(page == "Homepage"):
        st.header("About")
        st.write("Heal the world")
        viewLiveObjectDetect()
    elif(page == "StyleTransfer"):
        viewStyleTransfer()
    elif(page == "LiveVideoTransform"):
        viewWebRTCVideoTransform()    
    elif(page == "LiveObjectDetect"):
        viewLiveObjectDetect()    
    elif(page == "MediaStreaming"):
        viewMediaStreaming()            
    elif(page == "Segmentation"):        
        st.title("TBD")

    stn.style_transfer_file("./data/content_img_1.jpg", "./data/style_img_1.jpg", 0.5)        

if __name__ == "__main__":
    main()
