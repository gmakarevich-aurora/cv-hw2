import sys
import numpy as np
import os
import scipy.misc
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    #print ("W\n", W)
    U, S, V = np.linalg.svd(E)
    Q1 = np.dot(U, np.dot(W, V))
    Q2 = np.dot(U, np.dot(W.T, V))
    R1 = np.linalg.det(Q1) * Q1
    R2 = np.linalg.det(Q2) * Q2
    
    #print ("R1\n", R1)
    #print ("R2\n", R2)
    
    #print ("U\n", U)
    T1 = U[:, 2]
    #print ("T1\n", T1)
    T2 = -1 * T1
    
    RT = np.zeros((4, 3, 4))
    RT[0, :, :] = np.concatenate((R1, np.reshape(T1, (3, 1))), axis = 1)
    RT[1, :, :] = np.concatenate((R1, np.reshape(T2, (3, 1))), axis = 1)
    RT[2, :, :] = np.concatenate((R2, np.reshape(T1, (3, 1))), axis = 1)
    RT[3, :, :] = np.concatenate((R2, np.reshape(T2, (3, 1))), axis = 1)

    return RT
'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    #print ("Image points\n", image_points)
    #print ("camera_matrices\n", camera_matrices)
    A = np.zeros((2 * image_points.shape[0], 4))
    for idx in range(0, image_points.shape[0]):
        A[2*idx, :] = image_points[idx][0] * camera_matrices[idx, 2, :] - camera_matrices[idx, 0, :]
        A[2*idx + 1, :] = image_points[idx][1] * camera_matrices[idx, 2, :] - camera_matrices[idx, 1, :]
        
    #print ("A\n", A)
    U, S, V = np.linalg.svd(A)
    #print ("V\n", V)
    #print (V[-1, :])
    p3d = V[-1, [0, 1, 2]] / V[-1, 3]
    return p3d

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    point_3d_hg = np.append(point_3d, [1])
    # print ("point_3d_hg\n", point_3d_hg)
    error = np.zeros(2 * camera_matrices.shape[0])
    for idx in range(0, image_points.shape[0]):
        p = np.dot(camera_matrices[idx], point_3d_hg)
        # print ("p\n", p)
        error[2*idx] = p[0]/p[2] - image_points[idx][0]
        error[2*idx + 1] = p[1]/p[2] - image_points[idx][1]
    # print ("error\n", error)
    return error

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    point_3d_hg = np.append(point_3d, [1])
    jacobian_m = np.zeros((2 * camera_matrices.shape[0], 3))
    for idx in range(0, camera_matrices.shape[0]):
        p = np.dot(camera_matrices[idx], point_3d_hg)
        
        M = camera_matrices[idx]
        div = math.pow(p[2], 2)
        jacobian_m[2*idx] = np.array([
            M[0][0] * p[2] - p[0] * M[2][0],
            M[0][1] * p[2] - p[0] * M[2][1],
            M[0][2] * p[2] - p[0] * M[2][2]
        ])/div
        jacobian_m[2*idx + 1] = np.array([
            M[1][0] * p[2] - p[1] * M[2][0],
            M[1][1] * p[2] - p[1] * M[2][1],
            M[1][2] * p[2] - p[1] * M[2][2]
        ])/div
    
    return jacobian_m

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    p_estimate = linear_estimate_3d_point(image_points, camera_matrices)
    for it in range(0, 10):
        jacobian_m = jacobian(p_estimate, camera_matrices)
        error = reprojection_error(p_estimate, image_points, camera_matrices)
        p_estimate = p_estimate - np.linalg.inv(jacobian_m.T.dot(jacobian_m)).dot(jacobian_m.T).dot(error)
    return p_estimate

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    RT_estimates = estimate_initial_RT(E)
    camera_matrices = np.zeros((2, 3, 4))
    identity = np.concatenate((np.eye(3, 3), np.zeros((3, 1))), axis = 1)
    # print ("identity\n", identity)
    camera_matrices[0] = K.dot(identity)
    
    point_3d_estimate = np.zeros((RT_estimates.shape[0], image_points.shape[0], 3))
    
    good_points = np.zeros((RT_estimates.shape[0], 1))
    
    for rt_idx in range(0, RT_estimates.shape[0]):
        camera_matrices[1] = K.dot(RT_estimates[rt_idx])
        for p_idx in range(0, image_points.shape[0]):
                p_estimate = nonlinear_estimate_3d_point(image_points[p_idx], camera_matrices)
                point_3d_estimate[rt_idx][p_idx] = p_estimate
                # print ("p_estimate\n", p_estimate)
                if p_estimate[2] > 0:
                    p_estimate_hg = np.append(p_estimate, [1])
                    p_estimate_matrix_2 = RT_estimates[rt_idx].dot(p_estimate_hg)
                    # print ("p_estimate_matrix_2\n", p_estimate_matrix_2)
                    if p_estimate_matrix_2[2] > 0:
                        good_points[rt_idx][0] = good_points[rt_idx][0] + 1
                    
    best_idx = 0
    for rt_idx in range(0, good_points.shape[0]):
        if good_points[rt_idx][0] > good_points[best_idx][0]:
            best_idx = rt_idx
    
    return RT_estimates[best_idx]

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'))[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'))
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'))[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print '-' * 80
    print "Part A: Check your matrices against the example R,T"
    print '-' * 80
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = scipy.misc.imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print "Example RT:\n", example_RT
    estimated_RT = estimate_initial_RT(E)
    print
    print "Estimated RT:\n", estimated_RT

    # Part B: Determining the best linear estimate of a 3D point
    print '-' * 80
    print 'Part B: Check that the difference from expected point '
    print 'is near zero'
    print '-' * 80
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print "Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum()

    # Part C: Calculating the reprojection error and its Jacobian
    print '-' * 80
    print 'Part C: Check that the difference from expected error/Jacobian '
    print 'is near zero'
    print '-' * 80
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print "Error Difference: ", np.fabs(estimated_error - expected_error).sum()
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print "Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum()

    # Part D: Determining the best nonlinear estimate of a 3D point
    print '-' * 80
    print 'Part D: Check that the reprojection error from nonlinear method'
    print 'is lower than linear method'
    print '-' * 80
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print "Linear method error:", np.linalg.norm(error_linear)
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print "Nonlinear method error:", np.linalg.norm(error_nonlinear)

    # Part E: Determining the correct R, T from Essential Matrix
    print '-' * 80
    print "Part E: Check your matrix against the example R,T"
    print '-' * 80
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print "Example RT:\n", example_RT
    print
    print "Estimated RT:\n", estimated_RT

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print '-' * 80
    print 'Part F: Run the entire SFM pipeline'
    print '-' * 80
    frames = [0] * (len(image_paths) - 1)
    for i in xrange(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in xrange(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in xrange(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        c='k', depthshade=True, s=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    ax.view_init(-100, 90)

    plt.show()
