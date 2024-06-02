"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
from scipy import ndimage
# import itertools
# from scipy import stats
# import scipy
# import scipy.stats
# import time


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    raise NotImplementedError


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    # print(image.shape)
    # cv2.imshow('image', image)
    image_copy = np.copy(image)
    # cv2.imshow('gettiing corners', image_copy)
    # print("shape of the image = ", image_copy.shape)
    if len(image_copy.shape) < 3:
        max_y = image_copy.shape[0]-1
        max_x = image_copy.shape[1]-1
        corners = [(0, 0), (0, max_y), (max_x, 0), (max_x, max_y)]
    else:
        image_copy_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        # cv2.waitKey()
        max_y = image_copy_gray.shape[0]-1
        max_x = image_copy_gray.shape[1]-1
        corners = [(0, 0), (0, max_y), (max_x, 0), (max_x, max_y)]
    # print(corners)
    # cv2.waitKey()
    return corners
    #raise NotImplementedError


def find_markers(image, template=None, ishelper_for_part_4_and_5=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    # if np.median(image[:,:,0]) == np.median(image[:,:,1]) and np.median(image[:,:,0]) == np.median(image[:,:,2]):
    #     # cv2.imshow('def mono image', image)
    # else:
        # cv2.imshow('def color image', image)
    # cv2.imshow('def template', template)
    # cv2.imshow('starting image', image)
    # cv2.waitKey()
    # print("image shape 0 {}, 1 {}".format(image[0], image[1]))\
    # cv2.destroyAllWindows()
    # template_copy = np.copy(template)
    # image_copy = np.copy(image)
    # template_x_og = template_copy.shape[1]
    # template_y_og = template_copy.shape[0]
    ## grayscale for canny, which is good for edges
    if ishelper_for_part_4_and_5:
        print("running for video")
        match_search_results = []
        harris_search_results = []
        points2_match = []
        template_x_best_match = 0
        template_y_best_match = 0
        points2_harris = []
        template_x_best_harris = 0
        template_y_best_harris = 0
        kmeans_tuple_harris = []
        kmeans_tuple_match = []
        ###now checking color/brick image
        image_x = 0
        image_y = 0
        # for shape_range in np.linspace(0.125, 10, 16): #, 16:
        for shape_range in np.linspace(0.45, 3, 8):  # , 16:         # np.linspace(0.375, 12, 24): #, 16:
            for rot_values in np.arange(0, 60, 15):
                template_copy = np.copy(template)
                image_copy = np.copy(image)
                template_size = np.copy(template_copy)
                temp_dimensions = (int(template_size.shape[1] * shape_range), int(template_size.shape[0] * shape_range))
                ## change the template size
                template_size = cv2.resize(template_size, temp_dimensions, interpolation=cv2.INTER_AREA)
                temp_ratio_x = temp_dimensions[1] / image_copy.shape[1]
                temp_ratio_y = temp_dimensions[0] / image_copy.shape[0]
                ## change the rotation
                template_size_rot = ndimage.rotate(template_size, rot_values, reshape=False)
                if min(temp_ratio_x, temp_ratio_y) > 0.04 and max(temp_ratio_x, temp_ratio_y < 0.4):
                    if int(image_copy.shape[0] // 2) < template_size.shape[0] \
                            or int(image_copy.shape[1] // 2) < template_size.shape[1]:
                        break
                    # print("range {}\n template x {}".format(shape_range,template_size.shape[1]))
                    template_x = template_size.shape[1]
                    template_y = template_size.shape[0]
                    image_x = image_copy.shape[1]
                    image_y = image_copy.shape[0]
                    if min(np.max(image_copy[:, :, 0]), np.max(image_copy[:, :, 1]), np.max(image_copy[:, :, 2])) < 255:
                        template_gray = cv2.cvtColor(template_size_rot, cv2.COLOR_BGR2GRAY)
                        # template_gray = cv2.cvtColor(template_size, cv2.COLOR_BGR2GRAY)
                        gauss_blur_template = cv2.GaussianBlur(template_gray, (7, 7), 0)  # (3,3),0) # (9,9),0)
                        sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        template_gauss_blur_sharp = cv2.filter2D(gauss_blur_template, -1, sharpener_entry)
                        template_gauss_blur_sharp_float = np.float32(template_gauss_blur_sharp)
                        mask_template = np.zeros_like(template_size)
                        dest_template = cv2.cornerHarris(template_gauss_blur_sharp_float, blockSize=4, ksize=3, k=0.04)
                        # template_size[dest_template > 0.5 * dest_template.max()] = [0, 255, 0]
                        max_adj_temp = 0.7  # 0.65 #0.75 #0.85 #(0.95 - medblur/10) # 0.85 #0.75 #0.95 #35 #0.75 #0.5
                        ## make mask of just b/w for template, use pattern finding after applying same to image
                        mask_template[dest_template > max_adj_temp * dest_template.max()] = [255, 255, 255]
                        mask_template_1d = cv2.cvtColor(mask_template, cv2.COLOR_BGR2GRAY)
                        img_blur = cv2.medianBlur(image_copy, 9)  # medblur)
                        sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        img_med_blur_sharp = cv2.filter2D(img_blur, -1, sharpener_entry)
                        image_gauss_blur_sharp = cv2.GaussianBlur(img_med_blur_sharp, (5, 5), 0)
                        image_gauss_blur_sharp_gray = cv2.cvtColor(image_gauss_blur_sharp, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow("before the mask, dark: \n", image_gauss_blur_sharp_gray)
                        image_gauss_blur_sharp_float = np.float32(image_gauss_blur_sharp_gray)
                        # this is for the harris stuff
                        max_adj_image = 0.195  # 0.3 # 0.2
                        thresh_harris = 0.245  # 0.315 #0.275
                    else:
                        template_gray = cv2.cvtColor(template_size_rot, cv2.COLOR_BGR2GRAY)
                        # template_gray = cv2.cvtColor(template_size, cv2.COLOR_BGR2GRAY)
                        gauss_blur_template = cv2.GaussianBlur(template_gray, (5, 5), 0)  # (7,7)  # (3,3),0) # (9,9),0)
                        sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        template_gauss_blur_sharp = cv2.filter2D(gauss_blur_template, -1, sharpener_entry)
                        template_gauss_blur_sharp_float = np.float32(template_gauss_blur_sharp)
                        mask_template = np.zeros_like(template_size)
                        dest_template = cv2.cornerHarris(template_gauss_blur_sharp_float, blockSize=4, ksize=3, k=0.04)
                        # template_size[dest_template > 0.5 * dest_template.max()] = [0, 255, 0]
                        max_adj_temp = 0.6  # 0.65 #0.75 #0.85 #(0.95 - medblur/10) # 0.85 #0.75 #0.95 #35 #0.75 #0.5
                        ## make mask of just b/w for template, use pattern finding after applying same to image
                        mask_template[dest_template > max_adj_temp * dest_template.max()] = [255, 255, 255]
                        mask_template_1d = cv2.cvtColor(mask_template, cv2.COLOR_BGR2GRAY)
                        image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
                        img_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)  # (5,5),0)  # (9,9),0)
                        sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        img_med_blur_sharp = cv2.filter2D(img_blur, -1, sharpener_entry)
                        # this is for the harris stuff
                        image_gauss_blur_sharp_float = np.float32(img_med_blur_sharp)
                        max_adj_image = 0.175  # 0.195 #0.3 # 0.2 #0.225 #0.345 #0.35
                        thresh_harris = 0.3  # 0.325 #0.245 #0.29 #0.295 #0.28 #0.3 #0.4
                        # this works without messy background
                    mask_image = np.zeros_like(image_copy)
                    dest_image = cv2.cornerHarris(image_gauss_blur_sharp_float, blockSize=4, ksize=3, k=0.04)
                    mask_image[dest_image > max_adj_image * dest_image.max()] = [255, 255, 255]
                    mask_image_1d = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
                    search_harris = cv2.matchTemplate(mask_image_1d, mask_template_1d, cv2.TM_CCOEFF_NORMED)
                    (_harris, max_harris, _harris, loc_harris) = cv2.minMaxLoc(search_harris)
                    if len(harris_search_results) == 0 or max_harris > harris_search_results[0]:
                        harris_search_results = (max_harris, loc_harris)
                        template_x_best_harris = template_x
                        template_y_best_harris = template_y
                        points2_harris = np.argwhere(search_harris >= thresh_harris)
                    if len(points2_harris) < 5:
                        print("calling points match because len(points2_harris) = ",len(points2_harris))
                        template_gray = cv2.cvtColor(template_size_rot, cv2.COLOR_BGR2GRAY)
                        # template_gray = cv2.cvtColor(template_size, cv2.COLOR_BGR2GRAY)
                        gauss_blur_template = cv2.GaussianBlur(template_gray, (7, 7), 0)  # (3,3),0) # (9,9),0)
                        sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        template_gauss_blur_sharp = cv2.filter2D(gauss_blur_template, -1, sharpener_entry)
                        #
                        img_blur = cv2.medianBlur(image_copy, 5)  # medblur)
                        sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        img_med_blur_sharp = cv2.filter2D(img_blur, -1, sharpener_entry)
                        image_gauss_blur_sharp = cv2.GaussianBlur(img_med_blur_sharp, (5, 5), 0)
                        image_gauss_blur_sharp_gray = cv2.cvtColor(image_gauss_blur_sharp, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow("before the mask, dark: \n", image_gauss_blur_sharp_gray)
                        thresh_match = 0.5  # 0.8 #6 #8 #0.7
                        search_match = cv2.matchTemplate(image_gauss_blur_sharp_gray, template_gauss_blur_sharp,
                                                         cv2.TM_CCOEFF_NORMED)
                        (_match, max_match, _match, loc_match) = cv2.minMaxLoc(search_match)
                        if len(match_search_results) == 0 or max_match > match_search_results[0]:
                            match_search_results = (max_match, loc_match)
                            template_x_best_match = template_x
                            template_y_best_match = template_y
                            points2_match = np.argwhere(search_match >= thresh_match)
                    # end of for loop
        if len(points2_harris) > 0:
            points2_sort_harris = sorted(points2_harris, key=lambda x: x[0] + x[1])
            points2_sort_harris_copy = np.copy(points2_sort_harris)
            points3_harris = points2_sort_harris_copy + [template_x_best_harris // 2, template_y_best_harris // 2]
            x_harris = points3_harris[:, 1]
            y_harris = points3_harris[:, 0]
            xy_harris = np.dstack((x_harris, y_harris))
            XY_harris = np.float32(xy_harris[0])
            N_harris = 4
            if len(XY_harris) >= N_harris:
                criteria_match = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                ret_match, label_match, center_harris = cv2.kmeans(XY_harris, N_harris, None, criteria_match, 10
                                                                   , cv2.KMEANS_RANDOM_CENTERS)
                # convert centers to ints
                center_harris = center_harris.astype(int)
                # ref_harris_q1 = (0, 0)
                ref_harris_q1 = (image_x // 4, image_y // 4)
                kmeans_sort_harris_1 = sorted(center_harris, key=lambda x: (x[0] - ref_harris_q1[0]) ** 2
                                                                           + (x[1] - ref_harris_q1[1]) ** 2)
                ref_harris_q2 = (image_x // 4, (image_y // 2 + image_y // 4))
                kmeans_sort_harris_2 = sorted(center_harris, key=lambda x: (x[0] - ref_harris_q2[0]) ** 2
                                                                           + (x[1] - ref_harris_q2[1]) ** 2)
                ref_harris_q3 = ((image_x // 2 + image_x // 4), image_y // 4)
                kmeans_sort_harris_3 = sorted(center_harris, key=lambda x: (x[0] - ref_harris_q3[0]) ** 2
                                                                           + (x[1] - ref_harris_q3[1]) ** 2)
                ref_harris_q4 = ((image_x // 2 + image_x // 4), (image_y // 2 + image_y // 4))
                kmeans_sort_harris_4 = sorted(center_harris, key=lambda x: (x[0] - ref_harris_q4[0]) ** 2
                                                                           + (x[1] - ref_harris_q4[1]) ** 2)
                kmeans_sort_harris = np.vstack(
                    (kmeans_sort_harris_1[0], kmeans_sort_harris_2[0], kmeans_sort_harris_3[0]
                     , kmeans_sort_harris_4[0]))
                kmeans_tuple_harris = [tuple(b) for b in kmeans_sort_harris]
            else:
                kmeans_tuple_harris = []
        else:
            kmeans_tuple_harris = []
        if len(kmeans_tuple_harris) < 4:
            points2_sort_match = sorted(points2_match, key=lambda x: x[0] + x[1])
            points2_sort_copy_match = np.copy(points2_sort_match)
            print("what is points 2 {} \n what is template x best {} "
                  "\n what is template y best {}".format(points2_match,template_x_best_match,template_y_best_match))
            points3_match = points2_sort_copy_match + [template_x_best_match // 2, template_y_best_match // 2]
            x_match = points3_match[:, 1]
            y_match = points3_match[:, 0]
            # now for kmeans
            xy_match = np.dstack((x_match, y_match))
            XY_match = np.float32(xy_match[0])
            # print("xy_match",xy_match)
            N_match = 4
            if len(XY_match) >= N_match:
                criteria_match = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                ret_match, label_match, center_match = cv2.kmeans(XY_match, N_match, None, criteria_match
                                                                  , 10, cv2.KMEANS_RANDOM_CENTERS)
                # convert centers to ints
                center_match = center_match.astype(int)
                # ref_match_q1 = (0, 0)
                ref_match_q1 = (image_x // 4, image_y // 4)
                kmeans_sort_match_1 = sorted(center_match, key=lambda x: (x[0] - ref_match_q1[0]) ** 2
                                                                         + (x[1] - ref_match_q1[1]) ** 2)
                ref_match_q2 = (image_x // 4, (image_y // 2 + image_y // 4))
                kmeans_sort_match_2 = sorted(center_match, key=lambda x: (x[0] - ref_match_q2[0]) ** 2
                                                                         + (x[1] - ref_match_q2[1]) ** 2)
                ref_match_q3 = ((image_x // 2 + image_x // 4), image_y // 4)
                kmeans_sort_match_3 = sorted(center_match, key=lambda x: (x[0] - ref_match_q3[0]) ** 2
                                                                         + (x[1] - ref_match_q3[1]) ** 2)
                ref_match_q4 = ((image_x // 2 + image_x // 4), (image_y // 2 + image_y // 4))
                kmeans_sort_match_4 = sorted(center_match, key=lambda x: (x[0] - ref_match_q4[0]) ** 2
                                                                         + (x[1] - ref_match_q4[1]) ** 2)
                kmeans_sort_match = np.vstack((kmeans_sort_match_1[0], kmeans_sort_match_2[0], kmeans_sort_match_3[0]
                                               , kmeans_sort_match_4[0]))
                kmeans_tuple_match = [tuple(b) for b in kmeans_sort_match]
            target_location = kmeans_tuple_match
        else:
            target_location = kmeans_tuple_harris
    else:
        print("running for still")
        match_search_results = []
        harris_search_results = []
        points2_match = []
        template_x_best_match = 0
        template_y_best_match = 0
        points2_harris = []
        template_x_best_harris = 0
        template_y_best_harris = 0
        kmeans_tuple_harris = []
        kmeans_tuple_match = []
        if np.median(image[:,:,0]) == np.median(image[:,:,1]) and np.median(image[:,:,0]) == np.median(image[:,:,2]):
            template_copy = np.copy(template)
            image_copy = np.copy(image)
            template_x = template_copy.shape[1]
            template_y = template_copy.shape[0]
            image_x = image_copy.shape[1]
            image_y = image_copy.shape[0]
            template_gray = cv2.cvtColor(template_copy, cv2.COLOR_BGR2GRAY)
            gauss_blur_template = cv2.GaussianBlur(template_gray, (5, 5), 0) # (3,3),0) # (9,9),0)
            sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            template_gauss_blur_sharp = cv2.filter2D(gauss_blur_template, -1, sharpener_entry)
            # cv2.imshow("template_gauss_blur_sharp?", template_gauss_blur_sharp)
            # cv2.waitKey()
            image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            # this works without messy background
            gauss_blur_image = cv2.GaussianBlur(image_gray, (5, 5), 0) # (5,5),0)  # (9,9),0)
            # cv2.imshow("is gauss blur getting worse?", gauss_blur_image)
            # cv2.waitKey()
            image_gauss_blur_sharp = cv2.filter2D(gauss_blur_image, -1, sharpener_entry)
            ## try to make 2 match templates. first for match template with real image. second for harris corner.
            thresh_match = 0.8 #0.7
            # cv2.imshow('template_gauss_blur_sharp', template_gauss_blur_sharp)
            # cv2.waitKey()
            # cv2.imshow('image_gauss_blur_sharp', image_gauss_blur_sharp)
            # cv2.waitKey()
            search_match = cv2.matchTemplate(image_gauss_blur_sharp, template_gauss_blur_sharp, cv2.TM_CCOEFF_NORMED)
            (_match, max_match, _match, loc_match) = cv2.minMaxLoc(search_match)
            # print("match_search_results ", match_search_results)
            if len(match_search_results) == 0 or max_match > match_search_results[0]:
                match_search_results = (max_match, loc_match)
                template_x_best_match = template_x
                template_y_best_match = template_y
                # print("match_search_results 0", match_search_results[0])
                # print("what is max search_match ", np.max(search_match))
                # points_match = np.where(search_match >= thresh_match)
                # image_copy2 = np.copy(image)
                # image_copy2[points_match] = [0,0,255]
                # cv2.imshow('image_copy_match', image_copy2)
                # cv2.waitKey()
                # print("list of points: ", points_match)
                points2_match = np.argwhere(search_match >= thresh_match)
                # print("wahat does points2_match look like ",points2_match)
            ### above is just match below is harris and match
            ## make float for corner harris
            template_gauss_blur_sharp_float = np.float32(template_gauss_blur_sharp)
            # cv2.imshow("template_gauss_blur_sharp_float",template_gauss_blur_sharp_float)
            # cv2.waitKey()
            mask_template = np.zeros_like(template_copy)
            dest_template = cv2.cornerHarris(template_gauss_blur_sharp_float, blockSize=4, ksize=3, k=0.04)
            # template_copy[dest_template > 0.5 * dest_template.max()] = [0, 255, 0]
            max_adj_temp = 0.75 #0.5
            ## make mask of just b/w for template, use pattern finding after applying same to image
            mask_template[dest_template > max_adj_temp * dest_template.max()] = [255, 255, 255]
            mask_template_1d = cv2.cvtColor(mask_template, cv2.COLOR_BGR2GRAY)
            ## make float for corner harris
            image_gauss_blur_sharp_float = np.float32(image_gauss_blur_sharp)
            mask_image = np.zeros_like(image_copy)
            dest_image = cv2.cornerHarris(image_gauss_blur_sharp_float, blockSize=4, ksize=3, k=0.04)
            # image_copy[dest_image > 0.5 * dest_image.max()] = [0, 255, 0]
            # cv2.imshow('image_copy_harris', image_copy)
            # cv2.waitKey()
            max_adj_image = 0.35 # 0.4 # 0.35 # 0.5
            ## make mask of just b/w for image, use pattern finding after applying same to image
            mask_image[dest_image > max_adj_image * dest_image.max()] = [255, 255, 255]
            mask_image_1d = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            ##
            ## now for the harris corner search
            # cv2.imshow('mask_image_1d', mask_image_1d)
            # cv2.waitKey()
            # cv2.imshow('mask_template_1d', mask_template_1d)
            # cv2.waitKey()
            thresh_harris = 0.4 # 0.45 # 0.4 #0.75 #0.4 #0.5
            search_harris = cv2.matchTemplate(mask_image_1d, mask_template_1d, cv2.TM_CCOEFF_NORMED)
            (_harris, max_harris, _harris, loc_harris) = cv2.minMaxLoc(search_harris)
            # print("harris_search_results ", harris_search_results)
            if len(harris_search_results) == 0 or max_harris > harris_search_results[0]:
                harris_search_results = (max_harris, loc_harris)
                template_x_best_harris = template_x
                template_y_best_harris = template_y
                # print("harris_search_results 0", harris_search_results[0])
                # print("what is max search_harris ", np.max(search))
                # points_harris = np.where(search_harris >= thresh_harris)
                # image_copy3 = np.copy(image)
                # image_copy3[points_harris] = [255,0,255]
                # cv2.imshow('image_copy_harris', image_copy3)
                # cv2.waitKey()
                # print("list of points_harris: ", points_harris)
                points2_harris = np.argwhere(search_harris >= thresh_harris)
                # print("wahat does points 2 harris look like ",points2_harris)
            if len(points2_match) > 0:
                points2_sort_match = sorted(points2_match, key=lambda x: x[0] + x[1])
                points2_sort_copy_match = np.copy(points2_sort_match)
                points3_match = points2_sort_copy_match + [template_x_best_match // 2, template_y_best_match // 2]
                # print("points3_match ",points3_match)
                # image_copy2_match = np.copy(image)
                # image_copy2_match[points_match] = [255, 0, 255]
                # cv2.imshow('image_copy2_match', image_copy2_match)
                x_match = points3_match[:, 1]
                y_match = points3_match[:, 0]
                # print("all centers_match\n {}".format(tuple(zip(x_match, y_match))))
                # now for kmeans
                xy_match = np.dstack((x_match, y_match))
                # print("xy_match",xy_match)
                XY_match = np.float32(xy_match[0])
                # print("length XY_match",len(XY_match))
                # if num_times_called < 4:
                #     N_match = 4
                # else:
                #     N_match = 1
                N_match = 4
                if len(XY_match) >= N_match:
                    criteria_match = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    ret_match, label_match, center_match = cv2.kmeans(XY_match, N_match, None, criteria_match
                                                                      , 10, cv2.KMEANS_RANDOM_CENTERS)
                    # convert centers to ints
                    center_match = center_match.astype(int)
                    # ref_match_q1 = (0, 0)
                    ref_match_q1 = (image_x//4, image_y//4)
                    kmeans_sort_match_1 = sorted(center_match, key=lambda x: (x[0] - ref_match_q1[0]) ** 2
                                                                           + (x[1] - ref_match_q1[1]) ** 2)
                    ref_match_q2 = (image_x//4, (image_y//2 + image_y//4))
                    kmeans_sort_match_2 = sorted(center_match, key=lambda x: (x[0] - ref_match_q2[0]) ** 2
                                                                           + (x[1] - ref_match_q2[1]) ** 2)
                    ref_match_q3 = ((image_x//2 + image_x//4), image_y//4)
                    kmeans_sort_match_3 = sorted(center_match, key=lambda x: (x[0] - ref_match_q3[0]) ** 2
                                                                           + (x[1] - ref_match_q3[1]) ** 2)
                    ref_match_q4 = ((image_x//2 + image_x//4), (image_y//2 + image_y//4))
                    kmeans_sort_match_4 = sorted(center_match, key=lambda x: (x[0] - ref_match_q4[0]) ** 2
                                                                           + (x[1] - ref_match_q4[1]) ** 2)
                    kmeans_sort_match = np.vstack((kmeans_sort_match_1[0], kmeans_sort_match_2[0], kmeans_sort_match_3[0]
                                                   ,kmeans_sort_match_4[0]))
                    # print("list of points: 1 {}\n 2 {}\n 3 {}\n 4 {}".format(
                    #     kmeans_sort_match_1,kmeans_sort_match_2,kmeans_sort_match_3,kmeans_sort_match_4))
                    # print("does this kmeans match work ",kmeans_sort_match)
                    kmeans_tuple_match = [tuple(b) for b in kmeans_sort_match]
                    # print("kmeans_tuple match\n {}".format(kmeans_tuple_match))
                else:
                    kmeans_tuple_match = zip(x_match, y_match)
            else:
                ## columns are shape[1] and rows are [0]
                print("template match is none")
            if len(points2_harris) > 0:
                points2_sort_harris = sorted(points2_harris, key=lambda x: x[0] + x[1])
                points2_sort_harris_copy = np.copy(points2_sort_harris)
                points3_harris = points2_sort_harris_copy + [template_x_best_harris // 2, template_y_best_harris // 2]
                # print("points3_harris ",points3_harris)
                # image_copy2_harris = np.copy(image)
                # image_copy2_harris[points_harris] = [0, 0, 255]
                # cv2.imshow('image_copy2_harris', image_copy2_harris)
                x_harris = points3_harris[:, 1]
                y_harris = points3_harris[:, 0]
                # print("all centers_harris\n {}".format(tuple(zip(x_harris, y_harris))))
                # now for kmeans
                xy_harris = np.dstack((x_harris, y_harris))
                # print("xy_harris",xy_harris)
                XY_harris = np.float32(xy_harris[0])
                # print("length XY_harris",len(XY_harris))
                # if num_times_called < 4:
                #     N_harris = 4
                # else:
                #     N_harris = 1
                N_harris = 4
                if len(XY_harris) >= N_harris:
                    criteria_match = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    ret_match, label_match, center_harris = cv2.kmeans(XY_harris, N_harris, None, criteria_match, 10
                                                                      , cv2.KMEANS_RANDOM_CENTERS)
                    # convert centers to ints
                    center_harris = center_harris.astype(int)
                    # ref_harris_q1 = (0, 0)
                    ref_harris_q1 = (image_x//4, image_y//4)
                    kmeans_sort_harris_1 = sorted(center_harris, key=lambda x: (x[0] - ref_harris_q1[0]) ** 2
                                                                           + (x[1] - ref_harris_q1[1]) ** 2)
                    ref_harris_q2 = (image_x//4, (image_y//2 + image_y//4))
                    kmeans_sort_harris_2 = sorted(center_harris, key=lambda x: (x[0] - ref_harris_q2[0]) ** 2
                                                                           + (x[1] - ref_harris_q2[1]) ** 2)
                    ref_harris_q3 = ((image_x//2 + image_x//4), image_y//4)
                    kmeans_sort_harris_3 = sorted(center_harris, key=lambda x: (x[0] - ref_harris_q3[0]) ** 2
                                                                           + (x[1] - ref_harris_q3[1]) ** 2)
                    ref_harris_q4 = ((image_x//2 + image_x//4), (image_y//2 + image_y//4))
                    kmeans_sort_harris_4 = sorted(center_harris, key=lambda x: (x[0] - ref_harris_q4[0]) ** 2
                                                                           + (x[1] - ref_harris_q4[1]) ** 2)
                    kmeans_sort_harris = np.vstack((kmeans_sort_harris_1[0], kmeans_sort_harris_2[0], kmeans_sort_harris_3[0]
                                                   ,kmeans_sort_harris_4[0]))
                    # print("list of points: 1 {}\n 2 {}\n 3 {}\n 4 {}".format(
                    #     kmeans_sort_harris_1,kmeans_sort_harris_2,kmeans_sort_harris_3,kmeans_sort_harris_4))
                    # print("does this kmeans harris work ",kmeans_sort_harris)
                    kmeans_tuple_harris = [tuple(b) for b in kmeans_sort_harris]
                    # print("kmeans_tuple harris\n {}".format(kmeans_tuple_harris))
                else:
                    kmeans_tuple_harris = []
            else:
                print("harris is none")
            # target_location = []
            if harris_search_results[0] > 0.6 and len(points2_harris) > 0 and match_search_results[0] < 0.87:
                target_location = kmeans_tuple_harris
                # print("harris location {}".format(target_location))
            elif match_search_results[0] > 0.75 and len(points2_match) > 0:
                target_location = kmeans_tuple_match
                # print("match location {}".format(target_location))
            elif match_search_results[0] < 0.5 and len(points2_harris) > 0:
                target_location = kmeans_tuple_harris
                # print("harris location {}".format(target_location))
            elif harris_search_results[0] < 0.4 and len(points2_match) > 0:
                target_location = kmeans_tuple_match
                # print("match location {}".format(target_location))
            else:
                target_location = kmeans_tuple_harris
                # print("for mono color, you found nothing, using harris")
            # cv2.waitKey()
            # cv2.destroyAllWindows()
        ###now checking color/brick image
        else:
            image_x = 0
            image_y = 0
            # for shape_range in np.linspace(0.125, 10, 16): #, 16:
            # for shape_range in np.linspace(0.125, 4, 48): #, 16:         # np.linspace(0.375, 12, 24): #, 16:
            # for shape_range in np.linspace(0.125, 4, 16): #, 16:
            # for shape_range in np.linspace(0.125, 4, 8): #, 16:
            #     for rot_values in np.arange(0, 45, 5):
            # for shape_range in np.linspace(0.375, 4, 8):  # , 16:
            for shape_range in np.linspace(0.125, 4, 16):
                # for rot_values in np.arange(-60, 60, 10):
                # rot_iteratios = [1, 3, 9, 15, 30, 45, 60]
                # rot_values = itertools.cycle(rot_iteratios)
                template_copy = np.copy(template)
                image_copy = np.copy(image)
                template_size = np.copy(template_copy)
                temp_dimensions = (int(template_size.shape[1] * shape_range), int(template_size.shape[0] * shape_range))
                ## change the template size
                template_size = cv2.resize(template_size, temp_dimensions, interpolation=cv2.INTER_AREA)
                ## change the template rotation
                # print("this is the rotation cycle value ", next(rot_values))
                # template_size_rot = ndimage.rotate(template_size, next(rot_values), reshape=False)
                # print("this is the rotation cycle value ", rot_values)
                # template_size_rot = ndimage.rotate(template_size, rot_values, reshape=False)
                #
                temp_ratio_x = temp_dimensions[1]/image_copy.shape[1]
                temp_ratio_y = temp_dimensions[0]/image_copy.shape[0]
                if min(temp_ratio_x, temp_ratio_y) > 0.04 and max (temp_ratio_x, temp_ratio_y < 0.4):
                    if int(image_copy.shape[0]//2) < template_size.shape[0] \
                            or int(image_copy.shape[1]//2) < template_size.shape[1]:
                        break
                    # print("range {}\n template x {}".format(shape_range,template_size.shape[1]))
                    template_x = template_size.shape[1]
                    template_y = template_size.shape[0]
                    image_x = image_copy.shape[1]
                    image_y = image_copy.shape[0]
                    ## done with template processing, doing image
                        # for medblur in range(1, 13, 2):
                    # img_med_blur = cv2.medianBlur(image_copy, 13)
                    if min(np.max(image_copy[:,:,0]),np.max(image_copy[:,:,1]),np.max(image_copy[:,:,2])) < 255:
                        # print("darker image")
                        # template_gray = cv2.cvtColor(template_size_rot, cv2.COLOR_BGR2GRAY)
                        template_gray = cv2.cvtColor(template_size, cv2.COLOR_BGR2GRAY)
                        gauss_blur_template = cv2.GaussianBlur(template_gray, (7, 7), 0)  # (3,3),0) # (9,9),0)
                        sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        template_gauss_blur_sharp = cv2.filter2D(gauss_blur_template, -1, sharpener_entry)
                        template_gauss_blur_sharp_float = np.float32(template_gauss_blur_sharp)
                        mask_template = np.zeros_like(template_size)
                        dest_template = cv2.cornerHarris(template_gauss_blur_sharp_float, blockSize=4, ksize=3, k=0.04)
                        template_size[dest_template > 0.5 * dest_template.max()] = [0, 255, 0]
                        max_adj_temp = 0.7 #0.65 #0.75 #0.85 #(0.95 - medblur/10) # 0.85 #0.75 #0.95 #35 #0.75 #0.5
                        ## make mask of just b/w for template, use pattern finding after applying same to image
                        mask_template[dest_template > max_adj_temp * dest_template.max()] = [255, 255, 255]
                        # print("value for med blur ", medblur)
                        # cv2.waitKey()
                        # cv2.destroyAllWindows()
                        mask_template_1d = cv2.cvtColor(mask_template, cv2.COLOR_BGR2GRAY)
                        img_blur = cv2.medianBlur(image_copy, 9) #medblur)
                        sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        img_med_blur_sharp = cv2.filter2D(img_blur, -1, sharpener_entry)
                        image_gauss_blur_sharp = cv2.GaussianBlur(img_med_blur_sharp, (5, 5), 0)
                        image_gauss_blur_sharp_gray = cv2.cvtColor(image_gauss_blur_sharp, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow("before the mask, dark: \n", image_gauss_blur_sharp_gray)
                        image_gauss_blur_sharp_float = np.float32(image_gauss_blur_sharp_gray)
                        # this is for the harris stuff
                        max_adj_image = 0.195 #0.3 # 0.2
                        thresh_harris = 0.245 #0.315 #0.275
                        # this is for the match only stuff
                    else:
                        # print("lighter image")
                        # template_gray = cv2.cvtColor(template_size_rot, cv2.COLOR_BGR2GRAY)
                        template_gray = cv2.cvtColor(template_size, cv2.COLOR_BGR2GRAY)
                        gauss_blur_template = cv2.GaussianBlur(template_gray, (7, 7), 0) # (7,7)  # (3,3),0) # (9,9),0)
                        sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        template_gauss_blur_sharp = cv2.filter2D(gauss_blur_template, -1, sharpener_entry)
                        template_gauss_blur_sharp_float = np.float32(template_gauss_blur_sharp)
                        mask_template = np.zeros_like(template_size)
                        dest_template = cv2.cornerHarris(template_gauss_blur_sharp_float, blockSize=4, ksize=3, k=0.04)
                        # template_size[dest_template > 0.5 * dest_template.max()] = [0, 255, 0]
                        max_adj_temp = 0.6 # 0.7 # 0.8 #6 #0.4  #0.45 #5 #7 #6 #0.65 #0.75 #0.85 #(0.95 - medblur/10) # 0.85 #0.75 #0.95 #35 #0.75 #0.5
                        ## make mask of just b/w for template, use pattern finding after applying same to image
                        mask_template[dest_template > max_adj_temp * dest_template.max()] = [255, 255, 255]
                        # print("value for med blur ", medblur)
                        # cv2.waitKey()
                        # cv2.destroyAllWindows()
                        mask_template_1d = cv2.cvtColor(mask_template, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow('mask_template_1d', mask_template_1d)
                        # cv2.waitKey()
                        # img_blur = cv2.medianBlur(image_copy, 3) #medblur)
                        # sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        # img_med_blur_sharp = cv2.filter2D(img_blur, -1, sharpener_entry)
                        # image_gauss_blur_sharp = cv2.GaussianBlur(img_med_blur_sharp, (5, 5), 0)
                        # image_gauss_blur_sharp_gray = cv2.cvtColor(image_gauss_blur_sharp, cv2.COLOR_BGR2GRAY)
                        # img_blur = cv2.medianBlur(image_copy, 3) #medblur)
                        # image_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
                        # sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        # img_med_blur_sharp = cv2.filter2D(image_gray, -1, sharpener_entry)
                        # img_med_blur_sharp_gauss = cv2.GaussianBlur(img_med_blur_sharp, (3, 3), 0)  # (5,5),0)  # (9,9),0)
                        # this is for the harris stuff
                        image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
                        img_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)  # (5,5),0)  # (9,9),0)
                        sharpener_entry = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        img_med_blur_sharp = cv2.filter2D(img_blur, -1, sharpener_entry)
                        image_gauss_blur_sharp_float = np.float32(img_med_blur_sharp)
                        max_adj_image = 0.35 #0.19 # 0.195 #0.09 #08 #2 #0.08 #95 #0.25 #0.3 #0.35 #0.2 #0.3 #0.195 #0.225 #0.345 #0.3 # 0.195 #0.35 #0.195 #0.25 #0.195 #0.3 # 0.2 #0.225 #0.345 #0.35
                        thresh_harris = 0.3 #.35 # 0.325 #0.25 #4 #3 #0.195 #0.325 #0.29 #0.2 #0.295 #0.4  #0.2 #0.295 #0.4 #0.3 #0.325 #0.2 #0.4 #0.295 #0.325 #0.245 #0.29 #0.295 #0.28 #0.3 #0.4
                    ### above is just match below is harris and match
                    ## make float for corner harris
                    ## updated b/c i have other blur / guass function
                    mask_image = np.zeros_like(image_copy)
                    dest_image = cv2.cornerHarris(image_gauss_blur_sharp_float, blockSize=4, ksize=3, k=0.04)
                    # image_copy[dest_image > 0.1 * dest_image.max()] = [0, 255, 0]
                    # cv2.imshow('image_copy_harris before the best results', image_copy)
                    # cv2.waitKey()
                    mask_image[dest_image > max_adj_image * dest_image.max()] = [255, 255, 255]
                    mask_image_1d = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow('mask_image', mask_image)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                    ## now for the harris corner search
                    # thresh_harris = 0.45 #(0.3 + medblur/10) #0.35 #0.45 #S 0.75 # 0.5 # 0.3 #0.4 # 0.45 # 0.4 #0.75 #0.4 #0.5
                    search_harris = cv2.matchTemplate(mask_image_1d, mask_template_1d, cv2.TM_CCOEFF_NORMED)
                    (_harris, max_harris, _harris, loc_harris) = cv2.minMaxLoc(search_harris)
                    # print("harris_search_results ", harris_search_results)
                    if len(harris_search_results) == 0 or max_harris > harris_search_results[0]:
                        harris_search_results = (max_harris, loc_harris)
                        template_x_best_harris = template_x
                        template_y_best_harris = template_y
                        # print("harris_search_results 0", harris_search_results[0])
                        # print("what is max search_harris ", np.max(search))
                        # points_harris = np.where(search_harris >= thresh_harris)
                        # image_copy3 = np.copy(image)
                        # image_copy3[points_harris] = [255,0,255]
                        # print("list of points_harris: ", points_harris)
                        points2_harris = np.argwhere(search_harris >= thresh_harris)
                        # print("wahat does points 2 harris look like ",points2_harris)
                        # print("what is best med blur: ",medblur)
                # end of for loop
            if len(points2_harris) > 0:
                points2_sort_harris = sorted(points2_harris, key=lambda x: x[0] + x[1])
                points2_sort_harris_copy = np.copy(points2_sort_harris)
                points3_harris = points2_sort_harris_copy + [template_x_best_harris // 2, template_y_best_harris // 2]
                # print("points3_harris ",points3_harris)
                # image_copy2_harris = np.copy(image)
                # image_copy2_harris[points_harris] = [0, 0, 255]
                # cv2.imshow('image_copy2_harris - after best should be radius adjusted', image_copy2_harris)
                # cv2.waitKey()
                x_harris = points3_harris[:, 1]
                y_harris = points3_harris[:, 0]
                # print("all centers_harris\n {}".format(tuple(zip(x_harris, y_harris))))
                # now for kmeans
                xy_harris = np.dstack((x_harris, y_harris))
                # print("xy_harris",xy_harris)
                XY_harris = np.float32(xy_harris[0])
                # print("length XY_harris",len(XY_harris))
                # if num_times_called < 10:
                #     N_harris = 4
                # else:
                #     N_harris = 1
                N_harris = 4
                if len(XY_harris) >= N_harris:
                    criteria_match = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    ret_match, label_match, center_harris = cv2.kmeans(XY_harris, N_harris, None, criteria_match, 10
                                                                      , cv2.KMEANS_RANDOM_CENTERS)
                    # convert centers to ints
                    center_harris = center_harris.astype(int)
                    # ref_harris_q1 = (0, 0)
                    ref_harris_q1 = (image_x//4, image_y//4)
                    kmeans_sort_harris_1 = sorted(center_harris, key=lambda x: (x[0] - ref_harris_q1[0]) ** 2
                                                                           + (x[1] - ref_harris_q1[1]) ** 2)
                    ref_harris_q2 = (image_x//4, (image_y//2 + image_y//4))
                    kmeans_sort_harris_2 = sorted(center_harris, key=lambda x: (x[0] - ref_harris_q2[0]) ** 2
                                                                           + (x[1] - ref_harris_q2[1]) ** 2)
                    ref_harris_q3 = ((image_x//2 + image_x//4), image_y//4)
                    kmeans_sort_harris_3 = sorted(center_harris, key=lambda x: (x[0] - ref_harris_q3[0]) ** 2
                                                                           + (x[1] - ref_harris_q3[1]) ** 2)
                    ref_harris_q4 = ((image_x//2 + image_x//4), (image_y//2 + image_y//4))
                    kmeans_sort_harris_4 = sorted(center_harris, key=lambda x: (x[0] - ref_harris_q4[0]) ** 2
                                                                           + (x[1] - ref_harris_q4[1]) ** 2)
                    kmeans_sort_harris = np.vstack((kmeans_sort_harris_1[0], kmeans_sort_harris_2[0], kmeans_sort_harris_3[0]
                                                   ,kmeans_sort_harris_4[0]))
                    # print("list of points: 1 {}\n 2 {}\n 3 {}\n 4 {}".format(
                    #     kmeans_sort_harris_1,kmeans_sort_harris_2,kmeans_sort_harris_3,kmeans_sort_harris_4))
                    # print("does this kmeans harris work ",kmeans_sort_harris)
                    kmeans_tuple_harris = [tuple(b) for b in kmeans_sort_harris]
                    # print("kmeans_tuple harris\n {}".format(kmeans_tuple_harris))
                else:
                    kmeans_tuple_harris = []
            else:
                print("harris is none")
            # if harris_search_results[0] > 0.55 and len(points2_harris) > 0 and match_search_results[0] < 0.85:
            #     target_location = kmeans_tuple_harris
            #     # num_times_called += 1
            #     # print("harris location {}".format(target_location))
            # elif match_search_results[0] > 0.83 and len(points2_match) > 0:
            #     target_location = kmeans_tuple_match
            #     # num_times_called += 1
            #     # print("max location value: ", match_search_results[0])
            #     # print("match location {}".format(target_location))
            # elif match_search_results[0] < 0.65 and len(points2_harris) > 0:
            #     target_location = kmeans_tuple_harris
            #     # num_times_called += 1
            #     # print("harris location {}".format(target_location))
            # elif harris_search_results[0] < 0.4 and len(points2_match) > 0:
            #     target_location = kmeans_tuple_match
            #     # num_times_called += 1
            #     # print("match location {}".format(target_location))
            # else:
            target_location = kmeans_tuple_harris
                # num_times_called += 1
        # print("num_times_called: ", num_times_called)
    return target_location
    # raise NotImplementedError

def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    image_copy = np.copy(image)
    # cv2.imshow("image from find points",image_copy)
    # print("where are the points ",markers)
    # print("where are the points corners 1 {}"
    #             "\n 2 {}\n 3 {}"
    #             "\n 4 {}".format(markers[0],markers[1],markers[2],markers[3]))
    # cv2.waitKey()
    image_copy2 = cv2.line(image_copy, markers[0], markers[2], (249, 105, 14), thickness)
    image_copy3 = cv2.line(image_copy2, markers[0], markers[1], (249, 105, 14), thickness)
    image_copy4 = cv2.line(image_copy3, markers[2], markers[3], (249, 105, 14), thickness)
    image_copy5 = cv2.line(image_copy4, markers[1], markers[3], (249, 105, 14), thickness)
    # cv2.imshow("Image with box", image_copy5)
    # cv2.waitKey()
    return image_copy5
    #raise NotImplementedError


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    # print("what are sizes of images: \n source {} \n dest {}".format(imageA.shape, imageB.shape))
    dest_image = np.copy(imageB)
    dest_image2 = np.copy(imageB)
    src_image = np.copy(imageA)
    H_b4_inv = homography
    # invert the homography to go from destination values to source values
    # try without the inverse, then try with transpose
    H = np.linalg.inv(H_b4_inv)
    # trying without the inverse
    # H = homography
    # H = H_b4_inv
    ## referencing code guide endorsed by professor on piazza PS3-3
    ## https://stackoverflow.com/questions/46520123/how-do-i-use-opencvs-remap-function
    if len(dest_image.shape) < 3:
        dest_image_y, dest_image_x = dest_image.shape[:2]
    else:
        dest_image_gray = cv2.cvtColor(dest_image, cv2.COLOR_BGR2GRAY)
        # cv2.waitKey()
        dest_image_y, dest_image_x = dest_image_gray.shape[:2]
    index_dest_y, index_dest_x = np.indices((dest_image_y, dest_image_x), dtype=np.float32)
    # reshape_dest_y = index_dest_y.reshape(-1)
    # reshape_dest_x = index_dest_x.reshape(-1)
    # matrix_dest_index = np.array([reshape_dest_x, reshape_dest_y, np.ones_like(index_dest_x).reshape(-1)])
    matrix_dest_index = np.array([index_dest_x.ravel(), index_dest_y.ravel(), np.ones_like(index_dest_x).ravel()])
    map_dest_to_source_ind = H.dot(matrix_dest_index)
    map_dest_to_source_x2, map_dest_to_source_y2 = map_dest_to_source_ind[:-1] / map_dest_to_source_ind[-1]
    map_dest_to_source_x2 = map_dest_to_source_x2.reshape(dest_image_y, dest_image_x).astype(np.float32)
    map_dest_to_source_y2 = map_dest_to_source_y2.reshape(dest_image_y, dest_image_x).astype(np.float32)
    # cv2.remap(src=src_image, map1=map_dest_to_source_x2, map2=map_dest_to_source_y2, interpolation=cv2.INTER_LINEAR  #, interpolation=cv2.INTER_LINEAR
    #           , dst=dest_image2, borderMode=cv2.BORDER_TRANSPARENT) #, borderValue=0)
    cv2.remap(src_image, map_dest_to_source_x2, map_dest_to_source_y2, cv2.INTER_LINEAR, dest_image2
              , cv2.BORDER_TRANSPARENT)
    #           #, borderMode=cv2.BORDER_TRANSPARENT) #, borderValue=0)
    # dest_image2 = cv2.remap(src=src_image, map1=map_dest_to_source_x2, map2=map_dest_to_source_y2
    #           , interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT, borderValue=0)
    #
        #, borderValue=0, borderMode=cv2.BORDER_TRANSPARENT) #cv2.BORDER_REFLECT)
    # source_on_dest = cv2.addWeighted(dest_image, 0.15, dest_image2, 0.85, 0)
    # cv2.imshow('source_on_dest.png', source_on_dest)
    # cv2.waitKey()
    #raise NotImplementedError
    return dest_image2 #source_on_dest


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    # tryin svd, not sure why least squares is suggested.
    # declare martix for H - the homography
    # save points for the next one
    myH = np.zeros((3,3))
    myH2 = np.zeros((3,3))
    # lets have all the points
    # print("should i not use tuples? \n", src_points[1])
    x1s, y1s = src_points[0]
    x2s, y2s = src_points[1]
    # print("how does it look in the matrix: ", y2s)
    x3s, y3s = src_points[2]
    x4s, y4s = src_points[3]
    x1d, y1d = dst_points[0]
    x2d, y2d = dst_points[1]
    x3d, y3d = dst_points[2]
    x4d, y4d = dst_points[3]
    array_system_of_equations = np.array([
        [-x1s, -y1s, -1, 0, 0, 0, x1s*x1d, y1s*x1d, x1d],
        [0, 0, 0, -x1s, -y1s, -1, x1s*y1d, y1s*y1d, y1d],
        [-x2s, -y2s, -1, 0, 0, 0, x2s*x2d, y2s*x2d, x2d],
        [0, 0, 0, -x2s, -y2s, -1, x2s*y2d, y2s*y2d, y2d],
        [-x3s, -y3s, -1, 0, 0, 0, x3s*x3d, y3s*x3d, x3d],
        [0, 0, 0, -x3s, -y3s, -1, x3s*y3d, y3s*y3d, y3d],
        [-x4s, -y4s, -1, 0, 0, 0, x4s*x4d, y4s*x4d, x4d],
        [0, 0, 0, -x4s, -y4s, -1, x4s*y4d, y4s*y4d, y4d],
    ])
    [u, s, v] = np.linalg.svd(array_system_of_equations)
    # print("what does v look like: \n", v)
    myH = v[-1].reshape(3,3)
    other_H = myH/myH[2,2]   #myH/myH[-1]
    # # print("what is the difference between\n [2,2] {}\n [-1] {}".format(myH[2,2], myH[-1]))
    # # myHT = myH.transpose()
    # # other_H_T = myHT/myHT[2,2]   #myHT/myHT[-1]
    # myH2[0,0] = v[8, 0] # a
    # myH2[0,1] = v[8, 1] # b
    # myH2[0,2] = v[8, 2] # c
    # myH2[1,0] = v[8, 3] # d
    # myH2[1,1] = v[8, 4] # e
    # myH2[1,2] = v[8, 5] # f
    # myH2[2,0] = v[8, 6] # g
    # myH2[2,1] = v[8, 7] # h
    # myH2[2,2] = v[8, 8] # i
    # other_H2 = myH2/myH2[2,2]    #myH2/myH2[-1]
    # # myHT2 = myH2.transpose()
    # # other_H_T2 = myHT2/myHT2[2,2]    #myHT2/myHT2[-1]
    # list_src_points = [list(elem) for elem in src_points]
    # list_dst_points = [list(elem) for elem in dst_points]
    # print("what do the lists look like {} \n {}".format(list_src_points,list_dst_points))
    # array_src_points = np.array([list_src_points])
    # array_dst_points = np.array([list_dst_points])
    # ref_h, ref_status = cv2.findHomography(array_src_points, array_dst_points)
    # # print("this is my homography matrix: \n", myH)
    # print("this is my homography divided matrix: \n", other_H)
    # # # print("this is my homographyT matrix: \n", myHT)
    # # # print("this is my homographyT divided matrix: \n", other_H_T)
    # # # print("this is my homography assigned matrix: \n", myH2)
    # print("this is my homography divided assigned matrix: \n", other_H2)
    # # # print("this is my homographyT assigned matrix: \n", myHT2)
    # # # print("this is my homographyT divided assigned matrix: \n", other_H_T2)
    # print("this is opencv homography matrix: \n", ref_h)
    # cv2.waitKey()
    # # print("these are the destination points: \n", dst_points)
    return other_H #ref_h #other_H_T #ref_h #myH
    #raise NotImplementedError


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)
    # video = None

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    #raise NotImplementedError
    yield None


def find_aruco_markers(image, aruco_dict=cv2.aruco.DICT_5X5_50):
    """Finds all ArUco markers and their ID in a given image.

    Hint: you are free to use cv2.aruco module

    Args:
        image (numpy.array): image array.
        aruco_dict (integer): pre-defined ArUco marker dictionary enum.

        For aruco_dict, use cv2.aruco.DICT_5X5_50 for this assignment.
        To find the IDs of markers, use an appropriate function in cv2.aruco module.

    Returns:
        numpy.array: corner coordinate of detected ArUco marker
            in (X, 4, 2) dimension when X is number of detected markers
            and (4, 2) is each corner's x,y coordinate in the order of
            top-left, bottom-left, top-right, and bottom-right.
        List: list of detected ArUco marker IDs.
    """
    # image_aruco = np.float32(np.copy(image))
    image_copy = np.copy(image)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    # corners, ids, other = cv2.aruco.detectMarkers(image_copy, aruco_dict)
    aruco_param = cv2.aruco.DetectorParameters_create()
    corners2, ids2, other2 = cv2.aruco.detectMarkers(image=image_copy, dictionary=aruco_dict) #, parameters=aruco_param)
    # print("what corners are there: \n {}, ids {}".format(corners2,ids2))
    # print("what ids are there: \n", ids2)
    # cv2.imshow('image',image_copy)
    # cv2.waitKey()
    list_ids = []
    ordered_points_array = []
    # list_ids = sorted([ids2[0][0], ids2[1][0], ids2[2][0], ids2[3][0]])
    if ids2 is not None:
        for i in ids2:
            list_ids.append(i[0])
        list_ids.sort()
        # print(list_ids)
        # ap = []
        # ap_list = []
        ap_list_id = []
        for i in range(0, len(corners2)):
            if corners2[i] is not None:
                ap_temp = np.array(np.hstack(corners2[i])).astype(int)
                ap_list_temp = [tuple(b) for b in ap_temp]
                if ids2[i] is not None:
                    ap_list_id.append([len(ap_temp), ap_list_temp, ids2[i][0]])
        # print("ap_list_id ", ap_list_id)
        # if len(end) > 2:
        end = sorted((ap_list_id), key=lambda tup: tup[2])
        # print("end: ", end)
        # if len(ordered_points_array) > 2:
        ordered_points_array = [(x[0], x[1]) for x in end]
    # ordered_points_array = [(x[0], x[1]) for x in ap_list_id]
    # ap1 = np.array(np.hstack(corners2[0])).astype(int)
    # ap1_list = [tuple(b) for b in ap1]
    # ap1_list_id = [len(ap1), ap1_list, ids2[0][0]]
    # ap2 = np.array(np.hstack(corners2[1])).astype(int)
    # ap2_list = [tuple(b) for b in ap2]
    # ap2_list_id = [len(ap1), ap2_list, ids2[1][0]]
    # ap3 = np.array(np.hstack(corners2[2])).astype(int)
    # ap3_list = [tuple(b) for b in ap3]
    # ap3_list_id = [len(ap1), ap3_list, ids2[2][0]]
    # ap4 = np.array(np.hstack(corners2[3])).astype(int)
    # ap4_list = [tuple(b) for b in ap4]
    # ap4_list_id = [len(ap1), ap4_list, ids2[3][0]]
    # end2 = sorted((ap1_list_id, ap2_list_id, ap3_list_id, ap4_list_id), key=lambda tup: tup[2])
    # print("end: ",end2)
    # cv2.waitKey()
    # print("end: ",end)
    # cv2.imshow('image', image)
    # cv2.waitKey()
    # print(corners2, ids2)
    # print("count, points, id \n",ap1_list_id)
    # print("count, points, id \n",ap2_list_id)
    # print("count, points, id \n",ap3_list_id)
    # print("count, points, id \n",ap4_list_id)
    # ordered_points_array = [(x[0], x[1]) for x in end]
    # print(ordered_points_array)
    # cv2.imshow('image', image)
    # cv2.waitKey()
    return ordered_points_array, list_ids
    #raise NotImplementedError


def find_aruco_center(markers, ids):
    """Draw a bounding box of each marker in image. Also, put a marker ID
        on the top-left of each marker.

    Args:
        image (numpy.array): image array.
        markers (numpy.array): corner coordinate of detected ArUco marker
            in (X, 4, 2) dimension when X is number of detected markers
            and (4, 2) is each corner's x,y coordinate in the order of
            top-left, bottom-left, top-right, and bottom-right.
        ids (list): list of detected ArUco marker IDs.

    Returns:
        List: list of centers of ArUco markers. Each element needs to be
            (x, y) coordinate tuple.
    """
    # print("what do markers do like? \n", markers)
    ordered_points_array_no_qty = [(x[1]) for x in markers]
    # x = np.array(ordered_points_array_no_qty, dtype=int)
    # print(ordered_points_array_no_qty)
    # print(x)
    # corners_2_y = np.average(np.vstack((ordered_points_array_no_qty[0]))[:,0]).astype(int)
    # print(corners_2_y)
    x_avg_center = []
    y_avg_center = []
    x_median_center = []
    y_median_center = []
    for i in ordered_points_array_no_qty:
        x_avg_center.append(np.average(np.vstack((i))[:,0]).astype(int))
        y_avg_center.append(np.average(np.vstack((i))[:,1]).astype(int))
        #
        x_median_center.append(np.average(np.vstack((i))[:,0]).astype(int))
        y_median_center.append(np.average(np.vstack((i))[:,1]).astype(int))
    # print("x center: ", x_avg_center, x_median_center)
    # print("y center: ", y_avg_center, y_median_center)
    List_centers = []
    if len(x_avg_center) == len(y_avg_center):
        List_centers = [(x_avg_center[i], y_avg_center[i]) for i in range(0, len(x_avg_center))]
        # print(List_centers)
    else:
        List_centers = list(map(lambda x, y:(x,y), x_avg_center, y_avg_center))
    # print("all centers_match\n {}".format([tuple(zip(x_avg_center, y_avg_center))]))
    # corners_2 = sorted(np.hstack((corners2[1])).astype(int), key=lambda x: x[0] + x[1])
    # corners_1_med_x = np.median(np.array(corners_1)[:,1]).astype(int)
    # corners_1_med_y = np.median(np.array(corners_1)[:,0]).astype(int)
    # ids_2 = ids2[1]
    # corners_3 = sorted(np.hstack((corners2[1])).astype(int), key=lambda x: x[0] + x[1])
    # ids_3 = ids2[2]
    # corners_4 = sorted(np.hstack((corners2[1])).astype(int), key=lambda x: x[0] + x[1])
    # ids_4 = ids2[3]
    # print("the lists {}\n and {}\n and {}\n and {}".format(corners_1,corners_2,corners_3,corners_4))
    # print("the ids {} {} {} {} ".format(ids_1,ids_2,ids_3,ids_4))
    # print(np.median(np.array(corners_2)[:,1]))
    # print(np.mean(np.array(corners_2)[:,1]))
    return List_centers
    #raise NotImplementedError