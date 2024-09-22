import cv2
from pathlib import Path
from tqdm import tqdm

class FrameExtractor:
    def __init__(self, 
                 video_path, 
                 output_dir, 
                 num_frames,
                 frame_size=(640, 480)):
        # Convert the video_path and parent_dir 
        # to Path instances for easy manipulation.
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.num_frames = num_frames

        # Set the frame size for the extracted frames.
        self.frame_size = frame_size
        
        # Extract video file name without the 
        # extension.
        self.video_name = self.video_path.stem
        
        # Open the video file using OpenCV's 
        # VideoCapture.
        self.capture = cv2.VideoCapture(
            str(self.video_path)
        )
        
        # Create the output directory if it 
        # doesn't exist.
        self.output_dir.mkdir(
            parents=True, 
            exist_ok=True
        )

    def get_video_info(self):
        # Retrieve total frame count from the 
        # video.
        self.frame_count = int(self.capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        ))
        
        # Retrieve the frames per second (fps) 
        # of the video.
        self.fps = self.capture.get(
            cv2.CAP_PROP_FPS
        )
        
        # Calculate the total duration of the video.
        self.duration = self.frame_count / self.fps

    def extract_frames(self):
        # Calculate the interval to extract frames 
        # equally over the video duration.
        interval = self.frame_count // self.num_frames

        # Counter for the number of extracted frames
        frames_extracted = 0  
        
        # Loop through the video frame by frame.
        for i in tqdm(range(self.frame_count), unit=" frame"):
            # Read a frame from the video
            ret, frame = self.capture.read()
            
            # Break the loop if the frame 
            # couldn't be read (end of video).
            if not ret:
                break
            
            # Extract frames at equal intervals.
            if i % interval == 0:
                # Name the frame.
                frame_name = f"{self.video_name}-{frames_extracted + 1}.jpg"
                # Create full path using pathlib.
                frame_path = self.output_dir / frame_name
                
                # Save the frame as an image.
                resized_frame = cv2.resize(
                    frame, self.frame_size
                )
                cv2.imwrite(str(frame_path), resized_frame)
                
                # Increment the extracted frames counter.
                frames_extracted += 1
            
            # Stop when the desired number of frames has 
            # been extracted.
            if frames_extracted >= self.num_frames:
                break

    def release_resources(self):
        # Release the VideoCapture resource when done.
        self.capture.release()

def main():
    # Get the path to the video and the number 
    # of frames to extract from the user.
    video_path = "./dataset/videos/bvr-2006_07.mp4"
    num_frames = 500

    # Specify the frame size for the extracted
    # frames. Default is (1080, 720).
    frame_size=(1080, 720)
    
    # Create the output directory path using pathlib.
    output_dir = Path("./dataset/images") / Path(video_path).stem
    
    # Instantiate the FrameExtractor class with the 
    # video path, output directory, and number of frames.
    frame_extractor = FrameExtractor(
        video_path=video_path, 
        output_dir=output_dir, 
        num_frames=num_frames
    )
    
    # Get video details like frame count and fps.
    frame_extractor.get_video_info()
    
    # Extract the frames at equal intervals.
    frame_extractor.extract_frames()
    
    # Release resources after the extraction process.
    frame_extractor.release_resources()
    
    # Notify the user of successful completion
    print(f"Frames successfully extracted and saved at: {output_dir}")

if __name__ == "__main__":
    main()
