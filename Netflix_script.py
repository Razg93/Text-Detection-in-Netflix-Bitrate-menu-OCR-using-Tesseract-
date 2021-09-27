import os, sys, subprocess, pkg_resources
from functools import wraps
from timeit import default_timer as timer

#from nacl import exceptions

def is_on_intel_network():
    # importing socket module
    import socket
    # getting the hostname by socket.gethostname() method
    hostname = socket.gethostname()
    # getting the IP address using socket.gethostbyname() method
    ip_address = socket.gethostbyname(hostname)
    # printing the hostname and ip_address
    print("Current hostname: {}".format(hostname))
    print("Current IP Address: {}".format(ip_address))

    # if this script run on Intel network
    return ip_address.startswith("10")


def install_python_libraries(package):
    args_list = [sys.executable, "-m", "pip", "install", package]
    # if this script run on Intel network
    if is_on_intel_network():
        args_list += ["--proxy", "http://proxy.jer.intel.com:911"]
    subprocess.call(args_list)


def check_and_install_libraries():
    # install external libraries for text detection and image processing
    for package in ['python-dateutil', 'Pillow', 'pywin32', 'pypiwin32', 'matplotlib', 'lxml',
                    'pyscreenshot', 'opencv-python', 'pytesseract', 'numpy', 'mss', 'selenium', 'pandas']:
        try:
            dist = pkg_resources.get_distribution(package)
            print('{} ({}) is installed'.format(dist.key, dist.version))
        except pkg_resources.DistributionNotFound:
            print('{} is NOT installed. Installing it'.format(package))
            install_python_libraries(package)


check_and_install_libraries()

import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None)


def execution_time(fn):
    """
    wrapper to measure  the execution time of a given function
    :param fn:
    :return:
    """
    @wraps(fn)
    def internal(*args, **kwargs):
        start = timer()
        res = fn(*args, **kwargs)
        end = timer()
        print("{}\nFunction: {},\texecution time is: {} s\n{}".format("=" * 20, fn.__name__, end - start, "=" * 20))
        return res

    return internal


def show_image(img):
    plt.imshow(img, "gray")
    plt.show()


def save_image(path, img):
    """
    save input image in input path
    :param path:
    :param img:
    :return:
    """
    # verify the path is a folder and exists
    # if os.path.isdir(path):
    cv2.imwrite(path, img)
    # else:
    #     error_print("Input path: [{}] is not folder or does not exists.".format(path))


def dilate_img(thresh_img):
    """
    Specify structure shape and kernel size.
    Kernel size increases or decreases the area
    of the rectangle to be detected.
    A smaller value like (2, 6) will detect
    each word instead of a line.
    """
    rect_kernel = np.ones((3, 20), np.uint8)
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh_img, rect_kernel)
    #show_image(dilation)
    return dilation


def find_lines(dilate_img, img):
    """
    This function get dilation image
     and return text using tesseract
     """
    res = img.copy()
    num_labels, labels = cv2.connectedComponents(dilate_img)
    # we start from 2 to avoid all picture
    # for i in range(2, num_labels):
    text_list = []
    for i in range(2, num_labels):
        i_im = labels == i
        img = plot_rec(i_im, res)
        # multiple pxile
        text = extract_text(img)
        text = text.strip()
        text_list.append(text)
    return text_list


def edge_detect(image_gray):
    """get gray image and find the borders and cover it
    preprocess before dilate image"""
    edges = cv2.Canny(image_gray, 50, 400, apertureSize = 3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 1000)
    res = image_gray.copy()

    try:
        for r_t in lines:
            rho = r_t[0, 0]
            theta = r_t[0, 1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 3840 * (-b))
            y1 = int(y0 + 2160 * (a))
            x2 = int(x0 - 3840 * (-b))
            y2 = int(y0 - 2160 * (a))
            res = cv2.line(res, (x1, y1), (x2, y2), (0, 255, 255), thickness=5)
    except Exception as e:
        print("Error while edge detection. Message: {}".format(str(e)))

    return res


def plot_rec(mask, res_im):
    """
    plot a rectengle around each word in res image
    using mask image of the word
    """
    print("res!!!!")
    xy = np.nonzero(mask)
    y = xy[0]
    x = xy[1]
    left = x.min()-3
    right = x.max()+3
    up = y.min()-3
    down = y.max()+3
    cropped = res_im[up:down, left:right]
    return cropped


def pre_process_image(image):
    """
    this function pre process the image
    we filter only the numbers part using colors
    converting the image to GRAY format
    then convert it to binary image
    :param image_path:
    :return:
    """
    # convert input image to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    return thresh_img


def extract_text(img):
    """
    install tesseract on local machine if needed
    process input image and extract the text
    :param img:
    :return: text from the image
    """
    import pytesseract
    # default installation location
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if not os.path.exists(tesseract_path):
        url = "https://github.com/UB-Mannheim/tesseract/wiki"
        print("{}\tPlease install tesseract on local machine."
              " You can download the file from: {}\t{}".format("*" * 10, url, "*" * 10))
        exit(-1)

    print("Start looking for text in image")
    # init pytesseract
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    # process the image, try to get numbers from image
    config = ("-l eng --oem 1 --psm 7")
    return pytesseract.image_to_string(img, config=config) #-c tessedit_char_whitelist=/"


def create_dict(lines,file_name):
    """
    convert lines to dictionary elements
    """
    keys = ['file','Playing bitrate (a/v)','Playing/Buffering vmaf','Buffering bitrate (a/v)','Buffer size in Bytes (a/v)','Buffer size in Bytes','Buffer size in Seconds (a/v)','Framerate',
            'Current Dropped Frames','Total Frames','Total Dropped Frames','Total Corrupted Frames','Total Frame Delay','Throughput']
    d = {}
    print(lines)
    d['file'] = file_name
    for line in lines:
        # if we don't have ':" in the line ignore it
        if line.find(":") < 0:
            continue
        # split the line using ':'
        items = line.strip().split(':')
        # if key is not in keys list ignore it
        if items[0] not in keys:
            print(line[0])
            print(keys)
            continue
        # parse value part
        new_value = None
        audio_value = None
        video_value = None
        if items[1].find("x") > 0:
            audio_value, video_value = extract_resulotion(items[1])
        elif items[1].find("/") > 0:
            audio_value, video_value = extract_double_value(items[1])
        else:
            new_value = items[1]

        # parse key part
        index = items[0].find("(")
        #     playing bitrate (a/v) -> playing_bitrate_audio , playing_bitrate_video
        if index > 0:
            # remove '()' part
            temp = items[0][:index]
            temp.replace(" ", "_")
            audio = "{}_audio".format(temp)
            video = "{}_video".format(temp)
            d[audio] = audio_value
            d[video] = video_value
        else:
            if len(items) == 1:
                d[items[0]] = ""

            else:
                d[items[0]] = items[1]
    return pd.Series(d)


def extract_resulotion(line):
    """
    extract only the resolution from the Playing bitrate value
    """
    index = line.find("(")
    # remove prefix
    line = line[index:]
    items = line.strip().replace("(", "").replace(")", "").split('x')
    if len(items) != 2:
        return
    items = tuple(items)
    return items


def extract_double_value(line):
    """
    extract and separate to 2 different values audio and video
    """
    index = line.find("(")
    if index > 0:
        # remove sofix
        line = line[:index]
    items = line.strip().split('/')
    if len(items) != 2:
        return
    items = tuple(items)
    return items


def process_image(input, is_path=True):
    if is_path:
        if not os.path.isfile(input):
            print("Cannot find image in path: {}".format(input))
            return
        # read image
        img = cv2.imread(input)
    else:
        img = input

    thresh_img = pre_process_image(img)
    no_edege = edge_detect(thresh_img)
    dilation = dilate_img(no_edege)
    lines = find_lines(dilation, img)
    d = create_dict(lines, input)
    return d


if __name__ == "__main__":
    df = pd.DataFrame()
    print("!!!!")
    path = './Netflix_Screen_Capture/*.jpeg'
    img_names = glob(path)
    for file in img_names:
        print('processing image on path: {}'.format(file))
        d = process_image(file)
        df = df.append(d, ignore_index=True)
    keys = ['file', 'Playing bitrate _audio', 'Playing bitrate _video', 'Playing/Buffering vmaf', 'Buffering bitrate _audio', 'Buffering bitrate _video',
            'Buffer size in Bytes _audio', 'Buffer size in Bytes _video', 'Buffer size in Bytes', 'Buffer size in Seconds _audio', 'Buffer size in Seconds _video',
            'Framerate', 'Current Dropped Frames', 'Total Frames', 'Total Dropped Frames', 'Total Corrupted Frames', 'Total Frame Delay', 'Throughput']
    df = df[keys]
    print('Save to CSV file')
    df.to_csv('NetflixLogs.csv', index=False)


