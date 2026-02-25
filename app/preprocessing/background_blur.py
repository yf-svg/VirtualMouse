import cv2

def blur_background(frame, roi=None, blur_strength=25):
    """
    Blurs the background of the frame, keeping the ROI (region of interest) sharp.
    If roi is None, blurs the entire frame.
    roi: (x, y, w, h) in pixel coordinates
    blur_strength: odd integer for Gaussian blur
    """
    if blur_strength % 2 == 0:
        blur_strength += 1
    if roi is None:
        return cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
    x, y, w, h = roi
    mask = frame.copy()
    mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
    blurred[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
    return blurred
