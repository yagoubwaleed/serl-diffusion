# File taken mostly from Daniel Yang's code
import cv2
import imageio


class VideoRecorder:
    '''
    this video recorder like it's obs to come in HWC format!
    '''

    def __init__(self, render_size=256, fps=20):

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        if obs.shape[1] % 16 != 0:
            self.render_size = (obs.shape[1] // 16 + 1) * 16
        self.record(obs)

    def record(self, obs):
        frame = obs
        if frame.shape[1] % 16 != 0:
            # resize to multiple of 16
            frame = cv2.resize(
                obs,
                dsize=(self.render_size, self.render_size),
                interpolation=cv2.INTER_CUBIC
            )
        # not needed for metaworld frames
        # frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
        #                 dsize=(self.render_size, self.render_size),
        #                 interpolation=cv2.INTER_CUBIC)
        self.frames.append(frame)

    def save(self, path):
        imageio.mimsave(path, self.frames, fps=self.fps)
