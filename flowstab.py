# Luke Phillips
# lcp2@pdx.edu

import argparse
import numpy as np
import cv2
from scipy import signal
from scipy import ndimage


lk_params = dict(winSize = (15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

lk_params2 = dict(winSize = (15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                 flags = cv2.OPTFLOW_USE_INITIAL_FLOW)

def rich_features(img, divisions = 4):
    feature_params = dict(maxCorners = 2000//divisions**2,
                          qualityLevel = 0.02,
                          minDistance = 5,
                          blockSize = 7 )
    features = []
    segments = []
    h, w = img.shape
    count = 0
    for i in range(0, divisions):
        for j in range(0, divisions):
            roi = ((j * w // divisions, i * h // divisions), ((j+1) * w // divisions, (i+1) * h // divisions))
            mask = np.zeros_like(img)
            cv2.rectangle(mask, roi[0], roi[1], 255, -1)
            p = cv2.goodFeaturesToTrack(img, mask=mask, **feature_params)
            if not (p is None):
                features.append(p)
                segments.append((count, count+p.shape[0]))
                count += p.shape[0]
    return np.concatenate(features), segments

def mesh_coords(i, j, img):
    h, w = img.shape[:2]
    y = (i+0.5) * h / 16
    x = (j+0.5) * w / 16
    return np.asarray((x, y))

def flow_index(x, y, img):
    h, w = img.shape[:2]
    j = int(x * 16 / w - 0.5)
    i = int(y * 16 / h - 0.5)
    return i, j

def hsv_vis(flow, scale=8):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    r, theta = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = theta * 180 / np.pi / 2
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(r * scale, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def compute_meshflow(vid, args):
    vid_length = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    meshflow = np.zeros((int(vid_length), 16, 16, 2))

    _ret, frame = vid.read()
    frame_count = 0
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_gray = frame_gray.copy()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if not args.output_file is None:
        out = cv2.VideoWriter(args.output_file, fourcc, vid.get(cv2.CAP_PROP_FPS)//2, (int(frame.shape[1]*2), int(frame.shape[0])))
    else:
        out = None
    
    while(True):
        _ret, frame = vid.read()
        if frame is None:
            return meshflow
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_points, segments = rich_features(last_gray, args.divisions)
        new_points, st, _err = cv2.calcOpticalFlowPyrLK(last_gray, frame_gray, last_points, None, **lk_params)
        reverse_points, st, _err = cv2.calcOpticalFlowPyrLK(frame_gray, last_gray, new_points, None, **lk_params)
        d = abs(last_points-reverse_points).reshape(-1,2).max(-1)
        passed = np.logical_and(d < 1, (st.reshape(-1) > 0))
        global_homography, good = cv2.findHomography(last_points, new_points, cv2.RANSAC, 6)
        global_points = cv2.perspectiveTransform(last_points, global_homography)
        new_points = global_points.copy()
        new_points, _st, _err = cv2.calcOpticalFlowPyrLK(last_gray, frame_gray, last_points, new_points, **lk_params2)
        inlier_mask = np.zeros(last_points.shape[0], dtype=np.uint8)
        for segment in segments:
            ret_, good = cv2.findHomography(last_points[slice(*segment)], new_points[slice(*segment)], cv2.RANSAC, 2)
            inlier_mask[slice(*segment)] = good[:,0]
        global_homography, good = cv2.findHomography(last_points, new_points, cv2.RANSAC, 6)
        global_points = cv2.perspectiveTransform(last_points, global_homography)

        np.bitwise_and(inlier_mask, passed)
        np.bitwise_and(inlier_mask, good[:,0])
        #global_points = last_points, global_homography)
        #padded = np.vstack((last_points.reshape(-1,2).T, np.ones(last_points.shape[0])))
        #padded = np.dot(global_homography, padded)
        #global_points = (padded[:2] / padded[2]).T[:,np.newaxis,:]
        last_gray = frame_gray

        #meshflow = np.zeros((16, 16, 2))
        contributions = np.empty((16, 16, 2), dtype=object)
        for i in range(16):
            for j in range(16):
                for k in range(2):
                    contributions[i,j,k] = []

        for p0, p1 in zip(global_points.reshape(-1,2)[inlier_mask == 1], new_points.reshape(-1,2)[inlier_mask == 1]):
            vp = p1 - p0
            i0, j0 = flow_index(p1[0], p1[1], frame_gray)
            imin = max(i0-3, 0)
            imax = min(i0+4, 16)
            jmin = max(j0-3, 0)
            jmax = min(j0+4, 16)
            for i in range(imin, imax):
                for j in range(jmin, jmax):
                    x, y = tuple(mesh_coords(i, j, frame_gray))
                    px = p1[0]
                    py = p1[1]
                    rx = frame_gray.shape[1] / 16 * 3
                    ry = frame_gray.shape[0] / 16 * 3
                    test = (x - px)**2 / rx**2 + (y - py)**2 / ry**2
                    if test <= 1:
                        contributions[i,j,0].append(vp[0])
                        contributions[i,j,1].append(vp[1])
                        #cv2.circle(vis, tuple(np.int32(mesh_coords(i, j, vis))), 3, (255, 0, 255))
                    #cv2.circle(vis, tuple(np.int32(p1)), 3, (255, 255, 0))
        
        for i in range(16):
            for j in range(16):
                p0 = mesh_coords(i, j, frame_gray)
                p1 = cv2.perspectiveTransform(p0[np.newaxis, np.newaxis,:], global_homography).reshape(2)
                v = p1 - p0
                for k in range(2):
                    meshflow[frame_count,i,j,k] = v[k]
                    if len(contributions[i,j,k]) > 0:
                        meshflow[frame_count,i,j,k] += np.median(np.asarray(contributions[i,j,k]))

        meshflow[frame_count] = ndimage.median_filter(meshflow[frame_count], size=(3, 3, 1))

        vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        for i in range(0, 16):
            for j in range(0, 16):
                pt = mesh_coords(i, j, vis)
                cv2.circle(vis, tuple(np.int32(pt)), 3, (255, 0, 0))
                cv2.line(vis, tuple(np.int32(pt)), tuple(np.int32(pt + meshflow[frame_count,i,j])), (255, 0, 0))

        lines = np.hstack((last_points, new_points)).reshape(-1, 2, 2).astype(int)
        cv2.polylines(vis, lines[inlier_mask==1], 0, (0, 255, 0))
        cv2.polylines(vis, lines[inlier_mask==0], 0, (0, 0, 255))
        for point, glob, ok in zip(new_points.reshape(-1,2), global_points.reshape(-1,2), inlier_mask):
            if ok == 1:
                cv2.circle(vis, tuple(np.int32(point)), 2, (0, 255, 0))
            else:
                cv2.circle(vis, tuple(np.int32(point)), 2, (0, 0, 255))
            cv2.circle(vis, tuple(np.int32(glob)), 2, (255, 0, 255))

        meshflow_interp = cv2.resize(meshflow[frame_count], dsize=(frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        meshflow_vis = hsv_vis(meshflow_interp)
        cv2.imshow('hsv', meshflow_vis)
        cv2.imshow('points', vis)
        if frame_count == int(args.pvis):
            print("Frame {} printed".format(frame_count))
            cv2.imwrite('hsv.png', meshflow_vis)
            cv2.imwrite('points.png', vis)
        if not out is None:
            out.write(np.hstack((vis, meshflow_vis)))
        ch = cv2.waitKey(50)
        if ch == 27:
            break
        elif ch == ord('p'):
            cv2.imwrite('hsv.png', meshflow_vis)
            cv2.imwrite('points.png', vis)
            print("Frame {} printed".format(frame_count))


def main():
    # parse arguments
    argparser = argparse.ArgumentParser(description='CS410 Project')
    argparser.add_argument('video_file', help='Video File')
    argparser.add_argument('output_file', help='Output File', nargs='?', default=None)
    argparser.add_argument('-f', '--flow', help="Precalculated Meshflow file")
    argparser.add_argument('-s', '--scale', help="Gaussian kernel size", type=int, default=10)
    argparser.add_argument('-c', '--crop', help="Crop output", type=int, default=10)
    argparser.add_argument('-d', '--divisions', help="Feature detection divisions", type=int, default=1)
    argparser.add_argument('-v', '--visualize', help="Output visualization instead of result", action='store_true')
    argparser.add_argument('-p', '--pvis', help="Print the visualization of the specified tracking frame", default=None)
    args = argparser.parse_args()

    # open input
    vid = cv2.VideoCapture(args.video_file)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.flow is None:
        meshflow = compute_meshflow(vid, args)
        cv2.destroyAllWindows()
        if not meshflow is None:
            np.save(args.video_file, meshflow)
    else:
        meshflow = np.load(args.flow)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if not args.output_file is None:
            if args.visualize == False:
                out = cv2.VideoWriter(args.output_file, fourcc, vid.get(cv2.CAP_PROP_FPS), (width-2*args.crop, height-2*args.crop))
            else:
                out = cv2.VideoWriter(args.output_file, fourcc, vid.get(cv2.CAP_PROP_FPS), (width*2, height*2))
        else:
            out = None

        virt_paths = np.cumsum(meshflow, axis=0)
        lowpassed = ndimage.filters.gaussian_filter1d(virt_paths, args.scale, axis=0, mode='mirror')
        lowpassed2 = ndimage.filters.gaussian_filter1d(meshflow, args.scale, axis=0, mode='mirror')
        diff = virt_paths - lowpassed

        for flow, filtered, diff in zip(meshflow, lowpassed2, diff):
            filtered_interp = cv2.resize(filtered, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
            filtered_vis = hsv_vis(filtered_interp, 32)
            diff_interp = cv2.resize(diff, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
            diff_vis = hsv_vis(diff_interp)

            _ret, frame = vid.read()
            warp = np.float32(diff_interp)
            warp[...,0] += np.arange(width)
            warp[...,1] += np.arange(height)[:,np.newaxis]
            warped = cv2.remap(frame, warp, None, cv2.INTER_LINEAR)

            cv2.imshow('diff', diff_vis)
            cv2.imshow('source', frame[args.crop:-args.crop,args.crop:-args.crop])
            cv2.imshow('warp', warped[args.crop:-args.crop,args.crop:-args.crop])
            if not out is None:
                if args.visualize == False:
                    out.write(warped[args.crop:-args.crop,args.crop:-args.crop])
                else:
                    out.write(np.vstack((np.hstack((filtered_vis, diff_vis)),np.hstack((frame, warped)))))
            ch = cv2.waitKey(50)
            if ch == 27:
                break

        if not out is None:
            out.release()

    vid.release()
    cv2.destroyAllWindows()
        

if __name__ == '__main__':
    main()
