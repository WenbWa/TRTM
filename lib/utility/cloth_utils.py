
import torch
import cv2 as cv
import numpy as np
import torch.nn.functional as torchfun
from scipy import interpolate


# convert array to tensor with dtype=torch.float32
def to_tensor(array, dtype=torch.float32):
    return torch.tensor(array, dtype=dtype)

# convert array to numpy with dtype=np.float32
def to_numpy(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

# show single image
def show_image(img, name='image'):
    cv.namedWindow(name)
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# print iteration process bar in terminal
def show_process(current, total, prefix='Process', suffix='Complete', decimals=1, length=100, fill='█', printEnd="\r"):
    # start process
    if current == 0:
        print('\n========================== Process Started ==========================')
    # calculate process percent
    percent = ("{0:." + str(decimals) + "f}").format(100 * ((current + 1) / float(total)))
    # calculate process length
    filledLength = int(length * (current + 1) // total)
    # generate process bar
    bar = fill * filledLength + '-' * (length - filledLength)
    # print process bar
    print(f'{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # finish process
    if current + 1 == total:
        print('\n========================== Process Finished ==========================')


# ------------------- Cloth Data Utilities ------------------- #
# rotate image with angle
def rotate_image(image, angle, back_color=(255., 255., 255.)):
    # get image dimension
    h, w = image.shape[:2]
    # get rotation matrix with angle
    M = cv.getRotationMatrix2D([int(h / 2), int(w / 2)], angle, 1)
    # rotate img with angle
    image_rotate = cv.warpAffine(image, M, (h, w), borderValue=back_color)
    return image_rotate

# rotate mesh with angle
def rotate_mesh(mesh, angle):
    # get rotation matrix with angle
    radius = angle * np.pi / 180
    sn, cs = np.sin(radius), np.cos(radius)
    rotate_matrix = np.eye(3)
    rotate_matrix[0, :2] = [cs, -sn]
    rotate_matrix[1, :2] = [sn, cs]
    # rotate mesh with rotation matrix
    mesh_rotate = np.einsum('ij,kj->ki', rotate_matrix, mesh)
    return mesh_rotate

# rotate mesh index order with 90, 180, 270 angle
def rotate_mesh_index(mesh, angle, x_dimension=21, y_dimension=21):
    # rotate mesh index with angle 90, 180, 270
    angle = (angle // 90) * 90
    # construct rotation matrix
    radius = angle * np.pi / 180
    sn, cs = np.sin(radius), np.cos(radius)
    rotate_matrix = np.eye(3)
    rotate_matrix[0, :2] = [cs, -sn]
    rotate_matrix[1, :2] = [sn, cs]
    # get rotation matrix with angle
    rotate_matrix = rotate_matrix[0:2, 0:2].astype(int)
    # initialize mesh_rotate
    mesh_rotate = np.zeros(mesh.shape)
    # rotate mesh index order
    for idx in range(mesh.shape[0]):
        idx_rotate = convert_relative_position_to_vertex(np.dot(convert_vertex_to_relative_position(idx, x_dimension, y_dimension), rotate_matrix), x_dimension, y_dimension)
        mesh_rotate[idx] = mesh[idx_rotate]
    return mesh_rotate

# assign mesh index according to template mesh
def assign_mesh_index(pred_mesh, template_mesh):
    # rotate mesh index with 0, 90, 180, 270
    mesh_index_rotated = [rotate_mesh_index(pred_mesh, an) for an in [0, 90, 180, 270]]
    # get distance between rotated mesh and template mesh
    distance = [np.linalg.norm(mesh_index_rotated[idx] - template_mesh) for idx in range(len(mesh_index_rotated))]
    # assign sample mesh with minimum distance
    return mesh_index_rotated[np.argmin(distance)]

# noise image with different noise
def noise_image(img, noise):
    # get noisy image
    img_noise = img + noise
    # get cloth mask
    img_mask = img[:, :, 0].copy()
    img_mask[img_mask == 255] = 0
    img_mask[img_mask > 0] = 255
    # mask noisy cloth
    img_noise = cv.bitwise_and(img_noise, img_noise, mask=img_mask)
    img_noise[img_noise == 0] = 255
    return img_noise.astype(np.uint8)

# generate gaussian noise with mu, sigma, and img/frequency size
def gaussian_noise(img, mu=-2, sigma=3, fre=1):
    # generate gaussian noise with img/frequency size
    noise = np.random.normal(mu, sigma, size=(int(img.shape[0] / fre), int(img.shape[1] / fre))).astype(int)
    # interpolate noise to img size
    mx = np.linspace(0, img.shape[1], noise.shape[1])
    my = np.linspace(0, img.shape[0], noise.shape[0])
    f_noise = interpolate.interp2d(mx, my, noise, kind='linear')
    noise = f_noise(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    # return gaussian noise
    return np.dstack((noise, noise, noise))


# ------------------- Cloth Vertex Utilities ------------------- #

# get relative vertex position [(-10, 10), (-10, 10)], [(-10, 10), (-7, 7)] from n_vertex
def convert_vertex_to_relative_position(nvtx, x_dimension, y_dimension):
    row = nvtx // x_dimension
    column = nvtx % x_dimension
    return [int(-(x_dimension-1)/2 + column), int(-(y_dimension-1)/2 + row)]

# get n_vertex from relative vertex position [(-10, 10), (-10, 10)]
def convert_relative_position_to_vertex(position, x_dimension, y_dimension):
    row = position[1] + (y_dimension - 1) / 2
    column = position[0] + (x_dimension - 1) / 2
    return int(row * x_dimension + column)

# convert mesh position to mesh pixel
def convert_mesh_vertex_position_to_mesh_image_pixel(mesh_position, image, scale=240):
    # locate image center
    img_center = np.array([int(image.shape[0]/2), int(image.shape[1]/2)])
    # convert vertex position to image pixel
    mesh_pixel = mesh_position.copy()
    for nvtx in range(mesh_position.shape[0]):
        mesh_pixel[nvtx, :-1] = convert_image_pos_to_pixel_pos(mesh_position[nvtx, :-1] * scale, img_center)
    return mesh_pixel

# convert mesh position to group position according to template group
def convert_mesh_vertex_position_to_group_vertex_position(mesh_position, template_group):
    # initialize group position
    group_position = np.zeros((len(template_group), mesh_position.shape[1]))
    # grasp_position = np.zeros((len(template_group), mesh_position.shape[1]))
    # loop all group's center position
    for n_group in range(len(template_group)):
        index_group = template_group[n_group]
        index_group = index_group[index_group >= 0].astype(int)
        mesh_group = np.asarray(mesh_position[index_group])
        group_position[n_group] = np.mean(mesh_group, axis=0)
    return group_position


# convert pixel position (ph, pw) to image position (x, y)
def convert_image_pos_to_pixel_pos(image_pos, image_center):
    #    # get image center
    #    image_center = np.array([image.shape[0]/2, image.shape[1]/2])
    # calculate pixel position (pw, ph) from image position (x, y)
    pixel_pos = np.array([int(image_center[0] - image_pos[1]), int(image_center[1] + image_pos[0])])
    return pixel_pos

# get relative image position [(480, 240), (240, 480) from n_vertex
def convert_vertex_to_template_image(n_vertex, x_dimension, y_dimension, image, scale=24):
    px, py = convert_vertex_to_relative_position(n_vertex, x_dimension, y_dimension)
    return [int(image.shape[0]/2 - py*scale), int(image.shape[1]/2 + px*scale)]

# get flat distance between two vertices
def find_vertices_flat_distance(nvtx1, nvtx2, x_dimension, y_dimension):
    edge_length = 2 / (x_dimension - 1)
    x_distance = abs(nvtx1 % x_dimension - nvtx2 % x_dimension) * edge_length
    y_distance = abs(nvtx1 // x_dimension - nvtx2 // x_dimension) * edge_length
    flat_distance = (x_distance ** 2 + y_distance ** 2) ** 0.5
    return flat_distance


# ------------------- Cloth Visibility Utilities ------------------- #

# find points inside circle
def find_circle_inside_points(center, radius, points):
    # calculate distance between center and points, with radius
    distance = np.sqrt(np.sum((points - center) ** 2, axis=1)) - radius
    # find points inside circle
    inside_points = points[distance < 0]
    # find points index
    inside_points_index = np.where(distance < 0)[0]
    # return inside points (n, 2)
    return np.asarray(inside_points), list(inside_points_index)

# find pick lists from co_picks, overlap_range = 2 * grasp_range
def find_top_lists(mesh_position, co_picks, grasp_range=1, pick_single=True):
    # get dimension of cloth
    x_dimension = 21
    y_dimension = int(mesh_position.shape[0] / x_dimension)
    edge_length = 2 / (x_dimension - 1)
    # initialize pick lists of two hands
    pick_list = []
    pick_vtx1 = 0
    pick_vtx2 = 0

    # find cloth vertices near co_picks
    points_temp, pick_list1 = find_circle_inside_points(co_picks[0], grasp_range * edge_length, mesh_position[:, :2])
    points_temp, pick_list2 = find_circle_inside_points(co_picks[1], grasp_range * edge_length, mesh_position[:, :2])
    pick_list2 = [nvtx for nvtx in pick_list2 if nvtx not in pick_list1]
    pick_list1 = [int(nvtx) for nvtx in pick_list1]
    pick_list2 = [int(nvtx) for nvtx in pick_list2]
    # print('pick_lists:', pick_list1, pick_list2)

    # find the top-layer vertices from pick_list1
    if len(pick_list1) > 0:
        # find the top vertex
        pick_vtx1 = pick_list1[np.argmax(mesh_position[pick_list1, 2])]
        # find vertices around the top vertex
        temp_list = pick_list1
        pick_list1 = []
        for nvtx in temp_list:
            if find_vertices_flat_distance(pick_vtx1, nvtx, x_dimension, y_dimension) < 2 * grasp_range * edge_length:
                pick_list1.append(nvtx)
                pick_list.append(nvtx)
    # find the top-layer vertices from pick_list2
    if len(pick_list2) > 0:
        # find the top vertex
        pick_vtx2 = pick_list2[np.argmax(mesh_position[pick_list2, 2])]
        # find vertices around the top vertex
        temp_list = pick_list2
        pick_list2 = []
        for nvtx in temp_list:
            if find_vertices_flat_distance(pick_vtx2, nvtx, x_dimension, y_dimension) < 2 * grasp_range * edge_length:
                pick_list2.append(nvtx)
                pick_list.append(nvtx)

    # assert pick_vtx1 is the nearest top-layer vertex to co_pick
    if len(pick_list1) > 0:
        pick_vtx1 = pick_list1[np.argmin(np.linalg.norm(mesh_position[pick_list1, :2] - co_picks[0]))]
    # assert pick_vtx2 is the nearest top-layer vertex to co_pick
    if len(pick_list2) > 0:
        pick_vtx2 = pick_list2[np.argmin(np.linalg.norm(mesh_position[pick_list2, :2] - co_picks[1]))]

    # only manipulate with one cloth vertex
    if pick_single and len(pick_list1) > 0:
        pick_list1 = [pick_vtx1]
        pick_list = [pick_vtx1]
    if pick_single and len(pick_list2) > 0:
        pick_list2 = [pick_vtx2]
        pick_list.append(pick_vtx2)
    return pick_list1, pick_list2, pick_vtx1, pick_vtx2, pick_list

# detect whether a vertex is hidden inside of a cloth: hidden_range = 2/4 * grasp_range
def check_hidden_vertex(mesh_position, n_vertex, grasp_range=1, hidden_range=2):
    # get dimension of cloth
    x_dimension = 21
    y_dimension = int(mesh_position.shape[0] / x_dimension)
    edge_length = 2 / (x_dimension - 1)
    # assign pick position
    co_picks = np.zeros((2, 2))
    co_picks[0] = mesh_position[n_vertex, :2]
    co_picks[1] = mesh_position[n_vertex, :2]
    # find the top layer vertex at pick position
    pick_list1, pick_list2, pick_vtx1, pick_vtx2, pick_list = find_top_lists(mesh_position, co_picks, grasp_range)
    # detect overlap of n_vertex and pick_point1
    hidden = True
    if find_vertices_flat_distance(pick_vtx1, n_vertex, x_dimension, y_dimension) <= hidden_range * edge_length: hidden = False
    return hidden

# evaluate cloth visible vertices
def cloth_visible_vertices(mesh_position):
    # init np_visible for mesh vertices
    np_visible = np.ones(mesh_position.shape[0])
    # detect hidden vertices
    for nvtx in range(mesh_position.shape[0]):
        np_visible[nvtx] = 1 - check_hidden_vertex(mesh_position, nvtx)
    # return np_visible flag
    return np_visible

# evaluate cloth visible groups
def cloth_group_visible_value_flag(template_group_vertex, mesh_vertex_visible_flag, visible_treshold=15):
    # init group_visible_value
    group_visible_value = np.zeros(template_group_vertex.shape[0])
    for ntx in range(mesh_vertex_visible_flag.shape[0]):
        if mesh_vertex_visible_flag[ntx] == 1:
            for ng in range(template_group_vertex.shape[0]):
                if ntx in list(template_group_vertex[ng]): group_visible_value[ng] += 1
    # init group visible flag
    group_visible_flag = np.zeros(template_group_vertex.shape[0])
    for n_group in range(template_group_vertex.shape[0]):
        if group_visible_value[n_group] >= visible_treshold: group_visible_flag[n_group] = 1
    return group_visible_value, group_visible_flag


# ------------------- Cloth Manipulate Utilities ------------------- #

# dual arm target oriented querying of visible group pairs within search_pairs
# group_visible_value = [visible_value0, ...]; group_visible_flag = [visible_flag0, ...]
# search_pairs = [[search_list0], [...], ...]
def dual_arm_target_oriented_flipping(group_visible_value, group_visible_flag, search_pairs):
    # init search_pairs value and candidate
    search_pairs_value = []
    search_pairs_candidate = []
    # loop over all search pairs
    for n_list in range(len(search_pairs)):
        search_pairs_value.append([])
        search_pairs_candidate.append([])
        # append search pairs value
        for n_pair in range(len(search_pairs[n_list])):
            group_visible_pair_value = group_visible_value[search_pairs[n_list][n_pair]]
            group_visible_pair_flag = np.sum(group_visible_flag[search_pairs[n_list][n_pair]]) == 2
            search_pairs_value[-1].append(np.sum(group_visible_pair_value) * group_visible_pair_flag)
        # find search pairs candidate for each list
        if max(search_pairs_value[-1]) > 0: search_pairs_candidate[-1].append(search_pairs[n_list][np.argmax(search_pairs_value[-1])])

    # init target pair
    target_pair = []
    for n_cand in range(len(search_pairs_candidate)):
        if len(search_pairs_candidate[n_cand]) > 0:
            target_pair = search_pairs_candidate[n_cand][0]
            break
    return target_pair, search_pairs_candidate, search_pairs_value

# single arm dragging of visible group vertex to canonical target
# group_visible_value = [visible_value0, ...]; group_visible_flag = [visible_flag0, ...]
def single_arm_flat_dagging(pred_group_image, mesh_group_position, mesh_group_image_pixel, mesh_group_visible_flag, target_group_position):
    # copy mesh group image
    pred_drag_image = pred_group_image.copy()
    # find distance between mesh_group and target_group
    distance_group = np.linalg.norm(mesh_group_position[:, :-1] - target_group_position[:, :-1], axis=1)
    distance_group[mesh_group_visible_flag == 0] = 0
    drag_index = np.argmax(distance_group)
    # find target_group image_pixel
    target_group_image_pixel = convert_mesh_vertex_position_to_mesh_image_pixel(target_group_position, pred_drag_image)

    # paint target group positions, dragging vector
    for n_vtx in range(target_group_image_pixel.shape[0]):
        paint_circle(pred_drag_image, target_group_image_pixel[n_vtx, :-1], radius=5, color=(0, 0, 0), size=-1)
    # paint dragging vector
    paint_circle(pred_drag_image, mesh_group_image_pixel[drag_index, :-1], radius=20, color=(255, 255, 0), size=-1)
    paint_circle(pred_drag_image, target_group_image_pixel[drag_index, :-1], radius=20, color=(255, 255, 0), size=-1)
    paint_line(pred_drag_image, mesh_group_image_pixel[drag_index, :-1], target_group_image_pixel[drag_index, :-1], color=(255, 255, 0), size=20)
    return pred_drag_image, drag_index


## generate target-oriented manipulation policy, pred_mesh_image, pred_group_image
def manipulate_cloth_prediction(pred_mesh, real_depth, template_info, show=False):
    # init mesh_image, group_image, policy_image
    pred_mesh_image = real_depth.copy()
    pred_group_image = real_depth.copy()

    # get mesh vertex_position, visible_flag
    mesh_vertex_position = pred_mesh[:, :-1]
    mesh_vertex_visible_flag = pred_mesh[:, -1]
    mesh_vertex_edge_index = template_info['edge_idx']
    mesh_vertex_image_pixel = convert_mesh_vertex_position_to_mesh_image_pixel(mesh_vertex_position, real_depth)
    # paint visible and hidden mesh vertices and edges
    paint_mesh_vertex_edge(pred_mesh_image, mesh_vertex_image_pixel, mesh_vertex_visible_flag, mesh_vertex_edge_index, h_color=(0, 68, 255), h_size=(2, 3), v_color=(255, 68, 0), v_size=(2, 3))

    # paint template_image, obtain mesh_group_visible_value and mesh_group_visible_flag
    pred_template_image, mesh_group_visible_value, mesh_group_visible_flag = paint_template_group_visible(template_info['image'], template_info['group_vtx_idx'], pred_mesh[:, -1])
    # get mesh group_position
    mesh_group_position = convert_mesh_vertex_position_to_group_vertex_position(pred_mesh[:, :-1], template_info['group_vtx_idx'])
    mesh_group_image_pixel = convert_mesh_vertex_position_to_mesh_image_pixel(mesh_group_position, real_depth)
    mesh_group_edge_index = template_info['group_edge_idx']
    # paint visible and hidden group vertices and edges
    paint_mesh_vertex_edge(pred_group_image, mesh_group_image_pixel, mesh_group_visible_flag, mesh_group_edge_index, h_color=(0, 68, 255), h_size=(10, 16), v_color=(255, 68, 0), v_size=(12, 24))

    # init policy image as group image
    pred_policy_image = pred_group_image.copy()
    # generate and paint single-arm dragging policy
    pred_policy_image, drag_index = single_arm_flat_dagging(pred_policy_image, mesh_group_position, mesh_group_image_pixel, mesh_group_visible_flag, template_info['target_pos'])
    # generate dual-arm flipping policy
    flat_pair, flat_pair_candidates, flat_pair_values = dual_arm_target_oriented_flipping(mesh_group_visible_value, mesh_group_visible_flag, template_info['flat_pairs'])
    triangle_pair, triangle_pair_candidates, triangle_pair_values = dual_arm_target_oriented_flipping(mesh_group_visible_value, mesh_group_visible_flag, template_info['triangle_pairs'])
    rectangle_pair, rectangle_pair_candidates, rectangle_pair_values = dual_arm_target_oriented_flipping(mesh_group_visible_value, mesh_group_visible_flag, template_info['rectangle_pairs'])
    # paint dual-arm flipping policy
    paint_target_policy_region(pred_policy_image, [flat_pair, triangle_pair, rectangle_pair], mesh_group_image_pixel)
    # paint policy candidates
    paint_target_candidate_label(pred_template_image, [flat_pair_candidates, triangle_pair_candidates, rectangle_pair_candidates])

    # show processed images
    if show:
        image1 = np.concatenate((pred_group_image, pred_policy_image), axis=1)
        image2 = np.concatenate((pred_mesh_image, real_depth), axis=1)
        show_image(np.concatenate((image1, image2), axis=0))
    return [pred_template_image, pred_mesh_image, pred_group_image, pred_policy_image, [mesh_group_image_pixel, [flat_pair, triangle_pair, rectangle_pair]]]


# paint hidden and visible mesh vertices and edges within image
def paint_mesh_vertex_edge(image, mesh_image_pixel, mesh_visible_flag, mesh_edge_idx, h_color=(0, 68, 255), h_size=(10, 16), v_color=(255, 68, 0), v_size=(12, 24)):
    # paint hidden edges
    for n_edge in range(mesh_edge_idx.shape[0]):
        idx0 = int(mesh_edge_idx[n_edge][0])
        idx1 = int(mesh_edge_idx[n_edge][1])
        if mesh_visible_flag[idx0] == 0 or mesh_visible_flag[idx1] == 0:
            paint_line(image, mesh_image_pixel[idx0, :-1], mesh_image_pixel[idx1, :-1], color=h_color, size=h_size[0])
    # paint hidden vertices
    for n_vtx in range(mesh_image_pixel.shape[0]):
        if mesh_visible_flag[n_vtx] == 0:
            paint_circle(image, mesh_image_pixel[n_vtx, :-1], radius=h_size[1], color=h_color, size=-1)

    # paint visible edges
    for n_edge in range(mesh_edge_idx.shape[0]):
        idx0 = int(mesh_edge_idx[n_edge][0])
        idx1 = int(mesh_edge_idx[n_edge][1])
        if mesh_visible_flag[idx0] == 1 and mesh_visible_flag[idx1] == 1:
            paint_line(image, mesh_image_pixel[idx0, :-1], mesh_image_pixel[idx1, :-1], color=v_color, size=v_size[0])
    # paint visible vertices
    for n_vtx in range(mesh_image_pixel.shape[0]):
        if mesh_visible_flag[n_vtx] == 1:
            paint_circle(image, mesh_image_pixel[n_vtx, :-1], radius=v_size[1], color=v_color, size=-1)
    return image

# paint visible group within template_image
def paint_template_group_visible(template_image, template_group, mesh_vertex_visible, x_dimension=21, y_dimension=21, color=(0, 0, 255), radius=5, treshold=15):
    # get cloth dimension and template_img
    template_img = template_image.copy()
    # counter for visible group
    group_visible_value, group_visible_flag = cloth_group_visible_value_flag(template_group, mesh_vertex_visible, visible_treshold=treshold)

    # paint visible group value within template_image
    for ng in range(group_visible_value.shape[0]):
        point1 = convert_vertex_to_template_image(template_group[ng][0], x_dimension, y_dimension, template_img, scale=24)
        point2 = convert_vertex_to_template_image(template_group[ng][np.count_nonzero(template_group[ng] >= 0)-1], x_dimension, y_dimension, template_img, scale=24)
        cv.putText(template_img, str(int(group_visible_value[ng])), (int((point1[1]+point2[1])/2)-20, int((point1[0]+point2[0])/2)+8), cv.FONT_HERSHEY_SIMPLEX, 1, color, 4)
        cv.putText(template_img, str(int(ng)), (int((point1[1]+point2[1])/2)-10, int((point1[0]+point2[0])/2) + 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 4)

    # paint visible vertices within template_image
    for ntx in range(mesh_vertex_visible.shape[0]):
        if mesh_vertex_visible[ntx] == 1:
            point = convert_vertex_to_template_image(ntx, x_dimension, y_dimension, template_img, scale=24)
            cv.circle(template_img, (point[1], point[0]), radius, (0, 0, 0), -1)
    return template_img, group_visible_value, group_visible_flag

# paint target policy region at target image
def paint_target_policy_region(target_image, policy_pairs, group_image_pixel, radius=50, size=10):
    if len(policy_pairs[0]) > 0:
        paint_circle(target_image, group_image_pixel[policy_pairs[0][0]], radius=radius, color=(0, 255, 0), size=size)
        paint_circle(target_image, group_image_pixel[policy_pairs[0][1]], radius=radius, color=(0, 255, 0), size=size)
    if len(policy_pairs[1]) > 0:
        paint_triangle(target_image, group_image_pixel[policy_pairs[1][0]], radius=radius, color=(0, 0, 255), size=size)
        paint_triangle(target_image, group_image_pixel[policy_pairs[1][1]], radius=radius, color=(0, 0, 255), size=size)
    if len(policy_pairs[2]) > 0:
        paint_rectangle(target_image, group_image_pixel[policy_pairs[2][0]], radius, radius, color=(255, 0, 0), size=size)
        paint_rectangle(target_image, group_image_pixel[policy_pairs[2][1]], radius, radius, color=(255, 0, 0), size=size)
    return target_image

# paint target candidate label at template image
def paint_target_candidate_label(template_image, policy_candidates, base_color=(255, 255, 255), paint_color=(0, 0, 0)):
    template_image[0:100, :] = base_color
    cv.putText(template_image, 'Flat: ' + str(policy_candidates[0]), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, paint_color, 2)
    cv.putText(template_image, 'Triangle: ' + str(policy_candidates[1]), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, paint_color, 2)
    cv.putText(template_image, 'Rectangle: ' + str(policy_candidates[2]), (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, paint_color, 2)
    return template_image

# paint circle within image
def paint_circle(image, center, radius=4, color=(0, 255, 0), size=-1):
    cv.circle(image, (int(center[1]), int(center[0])), radius, color, size)
    return image

# paint rectangle within image
def paint_rectangle(image, center, dh, dw, color=(0, 255, 0), size=-1):
    cv.rectangle(image, (int(center[1]-dw), int(center[0]-dh)), (int(center[1]+dw), int(center[0]+dh)), color, size)
    return image

# paint rectangle within image
def paint_triangle(image, center, radius=10, color=(0, 255, 0), size=-1):
    p0 = [center[0]-radius/2, center[1]-radius*(3**0.5)/2]
    p1 = [center[0]-radius/2, center[1]+radius*(3**0.5)/2]
    p2 = [center[0]+radius, center[1]]
    paint_line(image, p0, p1, color=color, size=size)
    paint_line(image, p1, p2, color=color, size=size)
    paint_line(image, p2, p0, color=color, size=size)
    return image

# paint line within image
def paint_line(image, start, end, color=(0, 255, 0), size=10):
    cv.line(image, (int(start[1]), int(start[0])), (int(end[1]), int(end[0])), color, size)
    return image
