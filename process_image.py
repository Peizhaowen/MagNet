import cv2
import os
from skimage import io, transform, img_as_ubyte, color, img_as_float
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pandas as pd
from pandas import DataFrame
import multiprocessing
import shutil


# copy all files
def copy_dir(src_path, target_path):
    filelist_src = os.listdir(src_path)
    for file in filelist_src:
        src_path_read_new = os.path.join(os.path.abspath(src_path), file)
        target_path_write_new = os.path.join(os.path.abspath(target_path), file)
        if os.path.isdir(src_path_read_new):
            if not os.path.exists(target_path_write_new):
                os.mkdir(target_path_write_new)
            copy_dir(src_path_read_new, target_path_write_new)
        else:
            shutil.copy(src_path_read_new, target_path_write_new)


def preprocess_image(R):
    # initial file
    path = 'testing_set/{0}R'.format(R)

    # file for measuring
    measure_path = 'testing_set/{}R_measure'.format(R)
    if not os.path.exists(measure_path):
        os.mkdir(measure_path)
    copy_dir(path, measure_path)

    # rename: tif -> png, compressing to 1024
    print('******')
    print('rename and compress images for measurement', ' ***R: ', R)
    filelist = os.listdir(measure_path)
    for item in filelist:
        sub_filelist = os.listdir(os.path.abspath(measure_path) + '/' + item)
        i = 0
        for img in sub_filelist:
            if img.endswith('.tif'):
                src = os.path.join(os.path.abspath(measure_path + '/' + item), img)
                dst = os.path.join(os.path.abspath(measure_path + '/' + item), '' + str(i) + '.png')
            try:
                os.rename(src, dst)
            except:
                continue
            imgi = io.imread(dst)
            imgg = color.rgb2gray(imgi)
            imgg8 = img_as_ubyte(imgg)
            com_factor = max(imgi.shape[0], imgi.shape[1])/1024
            dsti = transform.resize(imgg8, (round(imgi.shape[0]/com_factor),
                                            round(imgi.shape[1]/com_factor)))
            io.imsave(dst, dsti)
            i = i + 1

    # file for Unet training
    print('******')
    print('create 512*512 images for training', ' ***R: ', R)
    train_path = 'testing_set/{}R_unet_train'.format(R)
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    copy_dir(measure_path, train_path)
    filelist = os.listdir(train_path)
    for item in filelist:
        sub_filelist = os.listdir(os.path.abspath(train_path) + '/' + item)
        i = 0
        for img in sub_filelist:
            img_name = os.path.join(os.path.abspath(train_path + '/' + item), img)
            imgi = io.imread(img_name)
            imgg = color.rgb2gray(imgi)
            imgg8 = img_as_ubyte(imgg)
            dsti = transform.resize(imgg8, (512, 512))
            io.imsave(img_name, dsti)
            i = i + 1


# create files for measuring
def edge_resize(R):
    print('******')
    print('resize edge images for measurement', ' ***R: ', R)
    measure_path = 'testing_set/{}R_measure'.format(R)
    train_path = 'testing_set/{}R_unet_train'.format(R)
    filelist = os.listdir(measure_path)
    for item in filelist:
        sub_filelist = os.listdir(os.path.abspath(measure_path) + '/' + item)
        tot_num = len(sub_filelist)
        for i in range(tot_num):
            edge_unet = os.path.join(os.path.abspath(
                train_path + '/' + item + '/' + str(i) + '_unet.png'))
            edge_resize = os.path.join(os.path.abspath(
                measure_path + '/' + item + '/' + str(i) + '_unet_resize.png'))
            img_name = os.path.join(os.path.abspath(
                measure_path + '/' + item + '/' + str(i) + '.png'))
            imgii = io.imread(img_name)
            imgi = io.imread(edge_unet)
            imgg = color.rgb2gray(imgi)
            imgg8 = img_as_ubyte(imgg)
            dsti = transform.resize(imgg8, (imgii.shape[0], imgii.shape[1]))
            io.imsave(edge_resize, dsti)


def remove_repetitive_contours(ct):
    num = np.size(ct)
    cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum = [], [], [], [], [], []
    for i in range(0, num):
        cnt = ct[i]
        rect = cv2.minAreaRect(cnt)
        len1 = round(rect[1][0])
        len2 = round(rect[1][1])
        cen_x = round(rect[0][0])
        cen_y = round(rect[0][1])
        if not ((cen_x in cenx_sum or (cen_x + 1) in cenx_sum or (cen_x - 1) in cenx_sum
                 or (cen_x + 2) in cenx_sum or (cen_x - 2) in cenx_sum) and
                (cen_y in ceny_sum or (cen_y + 1) in ceny_sum or (cen_y - 1) in ceny_sum
                 or (cen_y + 2) in ceny_sum or (cen_y - 2) in ceny_sum)) and (len1 != 0 and len2 != 0):
            len1_sum.append(len1)
            len2_sum.append(len2)
            cenx_sum.append(cen_x)
            ceny_sum.append(cen_y)
            rot_sum.append(rect[2])
            cnt_sum.append(cnt)
    return cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum


def edge_filter(cenx_list, ceny_list, len1_list, len2_list, rot_list, ct):
    list_size = np.size(cenx_list)
    cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum \
        = [], [], [], [], [], []
    for i in range(list_size):
        cnt = ct[i]
        edge_thre = 20
        if not (cenx_list[i] < edge_thre or cenx_list[i] > 1024 - edge_thre
                or ceny_list[i] < edge_thre or ceny_list[i] > 1024 - edge_thre):
            len1_sum.append(len1_list[i])
            len2_sum.append(len2_list[i])
            cenx_sum.append(cenx_list[i])
            ceny_sum.append(ceny_list[i])
            rot_sum.append(rot_list[i])
            cnt_sum.append(cnt)
    return cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum


def quartile_threshold(list, box_factor):
    l25 = np.percentile(list, 25)
    l75 = np.percentile(list, 75)
    l_iqr = l75 - l25
    lowout = l25 - box_factor * l_iqr
    upout = l75 + box_factor * l_iqr
    return lowout, upout


def size_filter(cenx_list, ceny_list, len1_list, len2_list, rot_list, ct, lb, bf):
    list_size = np.size(cenx_list)
    cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum = [], [], [], [], [], []
    for i in range(list_size):
        cnt = ct[i]
        len = max(len1_list[i], len2_list[i])
        wid = min(len1_list[i], len2_list[i])
        # if (len_lowout <= len) and (wid_lowout <= wid):
        if (lb <= len) and (lb <= wid):
            len1_sum.append(len1_list[i])
            len2_sum.append(len2_list[i])
            cenx_sum.append(cenx_list[i])
            ceny_sum.append(ceny_list[i])
            rot_sum.append(rot_list[i])
            cnt_sum.append(cnt)
    return cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum


def get_center_pixel(img_gray, cenx, ceny, len1, len2, rot, pixel_factor):
    cen_px_sum1 = []
    for k in range(round(len1 / pixel_factor)):
        for j in range(round(len2 / pixel_factor)):
            cen_px_sum1.append(img_gray[ceny + j, cenx + k])
            cen_px_sum1.append(img_gray[ceny - j, cenx - k])
            cen_px_sum1.append(img_gray[ceny + j, cenx - k])
            cen_px_sum1.append(img_gray[ceny - j, cenx + k])
    cen_px_mean = np.mean(cen_px_sum1)
    return cen_px_mean


def pixels_filter(img_gray, ct, cenx_list, ceny_list, len1_list,
                  len2_list, rot_list, pixel_factor, pixel_threshold):
    list_size = np.size(cenx_list)
    cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum = [], [], [], [], [], []
    for i in range(list_size):
        cnt = ct[i]
        cen_px_mean = get_center_pixel(img_gray, cenx_list[i], ceny_list[i], len1_list[i],
                                       len2_list[i], rot_list[i], pixel_factor)
        if cen_px_mean >= pixel_threshold:
            len1_sum.append(len1_list[i])
            len2_sum.append(len2_list[i])
            cenx_sum.append(cenx_list[i])
            ceny_sum.append(ceny_list[i])
            rot_sum.append(rot_list[i])
            cnt_sum.append(cnt)
    return cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum


def get_all_pixel(img_gray, cenx_list, ceny_list, len1_list,
                  len2_list, rot_list, pixel_factor):
    list_size = np.size(cenx_list)
    px_sum = []
    for i in range(list_size):
        cen_px_mean = get_center_pixel(img_gray, cenx_list[i], ceny_list[i], len1_list[i],
                                       len2_list[i], rot_list[i], pixel_factor)
        px_sum.append(cen_px_mean)
    return px_sum


def all_pixel_filter(img_gray, cenx_list, ceny_list, len1_list, len2_list,
                     rot_list, ct, pixel_factor, box_factor):
    list_size = np.size(cenx_list)
    cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum = [], [], [], [], [], []
    px_sum = get_all_pixel(img_gray, cenx_list, ceny_list,
                           len1_list, len2_list, rot_list, pixel_factor)
    px_lowout, px_upout = quartile_threshold(px_sum, box_factor)
    for i in range(list_size):
        cnt = ct[i]
        cen_px_mean = get_center_pixel(img_gray, cenx_list[i], ceny_list[i],
                                       len1_list[i], len2_list[i], rot_list[i], pixel_factor)
        print(cen_px_mean, px_upout)
        if cen_px_mean <= max(20, px_upout):
            len1_sum.append(len1_list[i])
            len2_sum.append(len2_list[i])
            cenx_sum.append(cenx_list[i])
            ceny_sum.append(ceny_list[i])
            rot_sum.append(rot_list[i])
            cnt_sum.append(cnt)
    return cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum


def distance_filter(cenx_list, ceny_list, len1_list, len2_list, rot_list, ct):
    list_size = np.size(cenx_list)
    cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum \
        = [], [], [], [], [], []
    distance_factor = 2
    for i in range(list_size):
        cnt = ct[i]
        width = min(len1_list[i], len2_list[i])
        cenn_list = []
        for cenn in range(list_size):
            d = ((cenx_list[cenn] - cenx_list[i]) ** 2
                 + (ceny_list[cenn] - ceny_list[i]) ** 2) ** 0.5
            if d < width / distance_factor:
                if len1_list[cenn] > len1_list[i] and len2_list[cenn] > len2_list[i]:
                    cenn_list.append(len1_list[cenn])
            if d >= width / distance_factor:
                cenn_list.append(len1_list[cenn])
        if len(cenn_list) == list_size - 1:
            len1_sum.append(len1_list[i])
            len2_sum.append(len2_list[i])
            cenx_sum.append(cenx_list[i])
            ceny_sum.append(ceny_list[i])
            rot_sum.append(rot_list[i])
            cnt_sum.append(cnt)
    return cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum


def crop_image(img_single_open, x, y, image_length, w, h):
    ex_factor = 5
    if ex_factor < y < w - ex_factor and ex_factor < x < h - ex_factor:
        cImg = img_single_open[y - ex_factor: y + image_length + ex_factor,
               x - ex_factor: x + image_length + ex_factor]
    else:
        cImg = img_single_open[y: y + image_length, x: x + image_length]
    return cImg


def extract_particle(file, n, m, cnt, imgi_gray):
    file_path = file + '/4classes_typical_{}'.format(n)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    img_name = file + '/' + str(n) + '.png'
    imgii = io.imread(img_name)

    img_single = np.ones((imgii.shape[0], imgii.shape[1]))
    for i in range(imgii.shape[1]):
        for j in range(imgii.shape[0]):
            dist = cv2.pointPolygonTest(cnt, (i, j), False)
            if dist == 0 or dist == 1:
                img_single[j, i] = imgi_gray[j, i]
            else:
                img_single[j, i] = 255
    result_name = file_path + '/{}_particle_{}.png'.format(n, m)
    cv2.imwrite(result_name, img_single)
    # resize image ******
    img_single_open = cv2.imread(result_name)
    kernel_length = 3
    gray = cv2.cvtColor(img_single_open, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel_length, kernel_length), 0)
    ret, single_gray = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)
    _, contours_single, _ = cv2.findContours(single_gray, cv2.RETR_CCOMP,
                                             cv2.CHAIN_APPROX_SIMPLE)
    resize_name = file_path + '/{}_particle_{}_resize.png'.format(n, m)
    print('Contour#:', m, ' Length of contours:', len(contours_single))
    if len(contours_single) > 2:
        wlist, hlist = [], []
        for ct in contours_single:
            rect = cv2.boundingRect(ct)
            [x, y, w, h] = rect
            if w < imgii.shape[0] or h < imgii.shape[1]:
                wlist.append(w)
                hlist.append(h)
        for ct in contours_single:
            rect = cv2.boundingRect(ct)
            [x, y, w, h] = rect
            image_length = max(w, h)
            if w == max(wlist) or h == max(hlist):
                cropImg = crop_image(img_single_open, x, y, image_length,
                                     imgii.shape[0], imgii.shape[1])
    else:
        if len(contours_single) == 1:
            rect = cv2.boundingRect(contours_single[0])
            [x, y, w, h] = rect
            image_length = max(w, h)
            cropImg = crop_image(img_single_open, x, y, image_length,
                                 imgii.shape[0], imgii.shape[1])
        else:
            for ct in contours_single:
                rect = cv2.boundingRect(ct)
                [x, y, w, h] = rect
                image_length = max(w, h)
                if w < imgii.shape[0] or h < imgii.shape[1]:
                    cropImg = crop_image(img_single_open, x, y, image_length,
                                         imgii.shape[0], imgii.shape[1])
    resize_single = cv2.GaussianBlur(cv2.resize(cropImg, (64, 64)),
                                     (kernel_length, kernel_length), 0)
    resize_gray = cv2.cvtColor(resize_single, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(resize_name, resize_gray)
    label = lenet_predict(resize_name)
    return label


def get_scale(scale, imgi, imgi_gray):
    edges = cv2.Canny(imgi, 40, 120)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    num = np.size(contours)
    for i in range(0, num):
        cnt = contours[i]
        rect = cv2.minAreaRect(cnt)
        cenx = rect[0][0]
        ceny = rect[0][1]
        len1 = rect[1][0]
        len2 = rect[1][1]
        length = max(len1, len2)
        width = min(len1, len2)
        rot = rect[2]
        if width != 0:
            elg = length / width
        else:
            elg = 0

        if rot == 0 and elg > 10 and length > 50:
            cen_pixel = get_center_pixel(imgi_gray, round(cenx), round(ceny),
                                         round(len1), round(len2), round(rot), 2)
            print('pixel:', cen_pixel)
            if cen_pixel < 3:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(imgi, [box], -1, (255, 0, 0), 3)
                scale_factor = float(scale) / length
    return scale_factor


def canny_identify(file, scale, n, lb, bf, ef, pf, pf_all, pt, bt):
    # unet edges ******
    file_edge = file + '/' + str(n) + '_unet_resize.png'
    img = cv2.imread(file_edge)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, gray = cv2.threshold(gray, bt, 255, cv2.THRESH_BINARY)
    # initial TEMs ******
    file_ini = file + '/' + str(n) + '.png'
    imgi = cv2.imread(file_ini)
    imgi_gray = cv2.cvtColor(imgi, cv2.COLOR_RGB2GRAY)
    # Canny edge detection ******
    edges = cv2.Canny(gray, 15, 45)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # filter
    cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum \
        = remove_repetitive_contours(contours)
    cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum \
        = edge_filter(cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum)
    # cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum \
    #     = pixels_filter(gray, cnt_sum, cenx_sum, ceny_sum,
    #                     len1_sum, len2_sum, rot_sum, pf, pt)
    cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum \
        = size_filter(cenx_sum, ceny_sum, len1_sum, len2_sum,
                      rot_sum, cnt_sum, lb, bf)
    cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum \
        = distance_filter(cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum)
    # cenx_sum, ceny_sum, len1_sum, len2_sum, rot_sum, cnt_sum \
    #     = all_pixel_filter(imgi_gray, cenx_sum, ceny_sum,
    #     len1_sum, len2_sum, rot_sum, cnt_sum, pf_all, bf)

    if not len(cenx_sum) == 0:
        print('TEM -', file, n)
        print("All contours:", np.size(contours))
        sf = get_scale(scale, imgi, imgi_gray)
        print(scale, sf)
        length_output, width_output, elg_output = [], [], []
        cen_x_output, cen_y_output, rot_output, label_output = [], [], [], []
        for m in range(len(cenx_sum)):
            cnt = cnt_sum[m]
            len1i = len1_sum[m] + ef
            len2i = len2_sum[m] + ef
            cen_xi = cenx_sum[m]
            cen_yi = ceny_sum[m]
            rot = rot_sum[m]
            rect1 = ((cen_xi, cen_yi), (len1i, len2i), rot)
            box = cv2.boxPoints(rect1)
            box = np.int0(box)
            label = extract_particle(file, n, m, cnt, imgi_gray)
            if label == 'bullet' or label == 'co' or label == 'prism':
                if label == 'bullet':
                    draw_color = (128, 0, 128)
                elif label == 'co':
                    draw_color = (0, 0, 255)
                else:
                    draw_color = (40, 255, 62)
                img_draw = cv2.drawContours(imgi, [box], -1, draw_color, 2)
                length_output.append(len1i)
                width_output.append(len2i)
                elg_output.append(max(len1i, len2i) / min(len1i, len2i))
                cen_x_output.append(cen_xi)
                cen_y_output.append(cen_yi)
                rot_output.append(rot)
                label_output.append(label)
            else:
                img_draw = imgi
        result_name = file + '/' + str(n) + '_process4.png'
        cv2.imwrite(result_name, img_draw)
        df: DataFrame = pd.DataFrame(columns=["center_x(nm)"])
        for i in range(len(length_output)):
            df.loc[i, 'center_x(nm)'] = cen_x_output[i] * sf
            df.loc[i, 'center_y(nm)'] = cen_y_output[i] * sf
            df.loc[i, 'width(nm)'] = length_output[i] * sf
            df.loc[i, 'height(nm)'] = width_output[i] * sf
            df.loc[i, 'elongation'] = elg_output[i]
            df.loc[i, 'angle'] = rot_output[i]
            df.loc[i, 'type'] = label_output[i]
        csv_name = file + '/' + str(n) + '_process4.csv'
        df.to_csv(csv_name, index_label="index")


def lenet_predict(file_name):
    model = lenet_model
    image = cv2.imread(file_name)
    original_image = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (64, 64))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image
    result = model.predict(image, batch_size=1)[0]
    Pr = np.max(result)
    label = np.where(result == Pr)[0]
    if label[0] == 0:
        label = 'error'
    elif label[0] == 1:
        label = 'co'
    elif label[0] == 2:
        label = 'prism'
    else:
        label = 'bullet'
    label2 = "[{}]: {:.2f}%".format(label, Pr * 100)

    # draw the label on the image
    output = imutils.resize(original_image, width=400)
    cv2.putText(output, label2, (10, 375), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    result_name = file_name + '_{}.png'.format(label)
    cv2.imwrite(result_name, output)
    return label


# lb: low boundary - min with lowout
# bf: box plot factor
# ef: expansion factor
# pf: pixel factor area of pixel calculated
# pt: pixel boundary
# bt: binary thrshold
def measure_file_prepare(R):
    global lenet_model
    # file for measuring
    path = 'testing_set/{}R'.format(R)
    measure_path = 'testing_set/{}R_measure'.format(R)
    filelist = os.listdir(measure_path)
    lenet_model = load_model('trained_model/LeNet_(64,200,300)_64×64.hdf5')
    for item in filelist:
        sub_filelist = os.listdir(os.path.abspath(path + '/' + item))
        tot_num = len(sub_filelist)
        print(tot_num)
        file_name = measure_path + '/' + item
        scale = item[:-2]
        for i in range(tot_num):
            p = multiprocessing.Process(target=canny_identify, args=(file_name, scale, i, 20, 2, 10, 10, 4, 250, 180))
            p.start()
            p.join()
            canny_identify(file_name, scale, i, 10, 2, 10, 10, 4, 250, 180)
