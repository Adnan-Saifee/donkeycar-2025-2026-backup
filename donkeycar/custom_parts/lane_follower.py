import cv2
import numpy as np
from simple_pid import PID
import logging

logger = logging.getLogger(__name__)

class LaneFollower:
    '''
    OpenCV based dual-line follower controller.
    This version looks for two yellow lines and centers the car between them.
    '''

    def __init__(self, pid, cfg):
        self.overlay_image = cfg.OVERLAY_IMAGE
        self.scan_y = cfg.SCAN_Y
        self.scan_height = cfg.SCAN_HEIGHT
        self.color_thr_low = np.asarray(cfg.COLOR_THRESHOLD_LOW)
        self.color_thr_hi = np.asarray(cfg.COLOR_THRESHOLD_HIGH)
        self.target_pixel = cfg.TARGET_PIXEL
        self.target_threshold = cfg.TARGET_THRESHOLD
        self.confidence_threshold = cfg.CONFIDENCE_THRESHOLD
        self.steering = 0.0
        self.throttle = cfg.THROTTLE_INITIAL
        self.delta_th = cfg.THROTTLE_STEP
        self.throttle_max = cfg.THROTTLE_MAX
        self.throttle_min = cfg.THROTTLE_MIN
        self.pid_st = pid
        self.current_lane_width = 0
        self.prev_lane_width = 0
        self.prev_left_peak = 0
        self.prev_right_peak = 0

    def get_i_color(self, cam_img):
        """
        Returns the left and right yellow line indices and confidence.
        Always finds the largest yellow spike in two separate histograms
        for the left and right halves separately.
        """
        iSlice = self.scan_y
        scan_line = cam_img[iSlice:iSlice + self.scan_height, :, :]

        # Convert to HSV and threshold for yellow
        img_hsv = cv2.cvtColor(scan_line, cv2.COLOR_RGB2HSV)

        # Make a binary mask for the colors in our range (black and white mask)
        mask = cv2.inRange(img_hsv, self.color_thr_low, self.color_thr_hi)

        # Compute histogram (hist is a vertical sum of all the pixels)
        hist = np.sum(mask, axis=0)
        width = hist.shape[0] # width by 1 np array
        mid = width // 2

        # Left half of hist
        left_half = hist[:mid]
        if np.any(left_half > 0):

            # Find max in left_half, element-wise comparision to make a boolean array(same shape), 
            # np.where() -> finds the indices of the Trues
            # the average is taken because multiple max values could be found, due
            # to the lane boundaries being many pixels wide. 
            left_peak = int(np.average(np.where(left_half == left_half.max())[0]))
        else:
            left_peak = None
        
        # Right half of hist
        right_half = hist[mid:]
        if np.any(right_half > 0):
            right_peak = int(np.average(np.where(right_half == right_half.max())[0])) + mid
        else:
            right_peak = None

        # Combine valid peaks
        peaks = []
        if left_peak is not None:
            peaks.append(left_peak)
        if right_peak is not None:
            peaks.append(right_peak)

        # Keeps track of the lane width per iteration, to use in case one of the lines are not found
        if left_peak is not None and right_peak is not None:
            self.current_lane_width = right_peak - left_peak
                    
        if self.current_lane_width is not None:
            self.prev_lane_width = self.current_lane_width
        if left_peak is not None:
            self.prev_left_peak = left_peak
        if right_peak is not None:
            self.prev_right_peak = right_peak
        

        # ChatGPT created this confidence equation ^_^ cuz i didn't know what it meant
        # Confidence = fraction of yellow pixels in the slice
        confidence = np.sum(mask) / (mask.shape[0] * mask.shape[1])

        return peaks, confidence, mask

    def run(self, cam_img):
        if cam_img is None:
            return 0, 0, False, None

        peaks, confidence, mask = self.get_i_color(cam_img)

        if not peaks:
            logger.info(f"No peak detected: confidence {confidence} < {self.confidence_threshold}")
            return self.steering, self.throttle, cam_img

        # If only one line found, use it(and self.current_lane_width) to calculate the center
        # ****POTENTIAL ISSUE: If miscalculated lanes register, self.current_lane_width can change significantly ****
        if len(peaks) == 1:
            img_width = cam_img.shape[0]
            if (img_width - peaks[0]) >= (img_width / 2):
                # Left line detected, assume right line out of bounds
                peaks.append(peaks[0] + self.current_lane_width)
            elif(img_width - peaks[0]) < (img_width / 2):
                # Right line detected, assume left line out of bounds
                value = peaks[0] - self.current_lane_width
                peaks.insert(0, value)

        # Take the midpoint between the two detected lines
        left_line, right_line = peaks[0], peaks[1]
        center = int((left_line + right_line) / 2)

        if self.target_pixel is None:
            self.target_pixel = center
            logger.info(f"Automatically chosen center position = {self.target_pixel}")

        if self.pid_st.setpoint != self.target_pixel:
            self.pid_st.setpoint = self.target_pixel

        if confidence >= self.confidence_threshold:
            # PID correction
            self.steering = self.pid_st(center)

            # Adjust throttle based on deviation
            if abs(center - self.target_pixel) > self.target_threshold:
                self.throttle = max(self.throttle - self.delta_th, self.throttle_min)
            else:
                self.throttle = min(self.throttle + self.delta_th, self.throttle_max)
        else:
            logger.info(f"Low confidence: {confidence:.4f}")

        if self.overlay_image:
            cam_img = self.overlay_display(cam_img, mask, peaks, confidence)

        # Currently hardcoded
        self.throttle = 0.07
        return self.steering, self.throttle , cam_img

    def overlay_display(self, cam_img, mask, peaks, confidence):
        mask_exp = np.stack((mask,) * 3, axis=-1)
        iSlice = self.scan_y
        
        img = np.copy(cam_img)
        img[iSlice : iSlice + self.scan_height, :, :] = mask_exp

        # Draw and display 2 lines at the detected peaks
        for p in peaks:
            cv2.line(img, (p, iSlice), (p, iSlice + self.scan_height), (255, 0, 0), 2)

        # Draw center if two lines found
        if len(peaks) == 2:
            center = int((peaks[0] + peaks[1]) / 2)
            cv2.line(img, (center, 0), (center, img.shape[0]), (0, 255, 0), 2)

        display_str = [
            f"STEERING:{self.steering:.2f}",
            f"THROTTLE:{self.throttle:.2f}",
            f"PEAKS:{peaks}",
            f"CONF:{confidence:.2f}"
        ]
        y = 10
        x = 10

        # Display all the strings in display_str
        for s in display_str:
            cv2.putText(img, s, color=(0, 0, 0), org=(x, y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4)
            y += 12

        if img is None:
            print("No overlay image returned")
        return img
