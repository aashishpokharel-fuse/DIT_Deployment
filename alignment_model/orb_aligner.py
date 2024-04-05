import cv2
import numpy as np


class OrbAligner(object):
    def __init__(self, good_match_percent=0.05, match_thresh=0.03):
        """Constructor
        Parameters
        ----------
        max_matches : int,
        Max features to be matched by the ORB detector, by default 15500
        good_match_percent : float, optional
        Max good matches, by default 0.05
        match_thresh : float, optional
        Max ratio of (num of matches / image height) below which the image
        is considered not to match the reference image, by default 0.03
        """
        self.GOOD_MATCH_PERCENT = good_match_percent
        self.MATCH_THRESH = match_thresh

    def get_good_matches(self, matches, ratio_thr):
        """Filter out the matches which is below ratio_thr param
        Parameters
        ----------
        matches : Matches made by the Matcher
        ratio_thr : Per of good matches
        Returns
        ------
        Filtered out good matches list
        """
        good_matches = []
        for m in matches:
            if m[1].distance != 0:
                ratio = m[0].distance / m[1].distance
                if ratio < ratio_thr:
                    good_matches.append(m[0])
        return good_matches

    def get_descriptor_matches(self, descriptors1, descriptors2, method='knn'):
        """Calculates best descriptor matches
        Parameters
        ----------
        descriptors1 : Numpy
            Descriptors of query image
        descriptors2 : Numpy
            Descriptors of train image
        method : str, optional
            , by default 'knn'
        Returns
        -------
        List
            list of good matches
        """
        if method == 'knn':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches01 = matcher.knnMatch(descriptors1, descriptors2, k=16)
            matches10 = matcher.knnMatch(descriptors2, descriptors1, k=16)
            good_matches01 = self.get_good_matches(matches01, 0.7)
            good_matches10 = self.get_good_matches(matches10, 0.7)
            good_matches10_ = {(m.trainIdx, m.queryIdx) for m in good_matches10}
            matches = [m for m in good_matches01 if (m.queryIdx, m.trainIdx) in good_matches10_]

        else:
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(descriptors1, descriptors2, None)

            # Sort matches by score
            matches.sort(key=lambda x: x.distance, reverse=False)

            # Remove not so good matches
            num_good_matches = int(len(matches) * self.GOOD_MATCH_PERCENT)
            matches = matches[:num_good_matches]

        return matches

    def align_image(self, image, ref_image, max_matches=15500):
        """Align or Register Image using ORB features matching and Homography
        # :param img: Image.Image object to be aligned to the member Reference Image
        Parameters
        ----------
        image : Numpy
            Query Image
        ref_image : Dict
            Information about Train Image
        Returns
        -------
        Bool, Numpy
            flag: [bool] True if align well else Fale
            h_matrix: [h_matrix] homography matrix that transforms the image
        """
        # ref_image = cv2.resize(ref_image, (image["width"], image["height"]),
        #                        interpolation=cv2.INTER_CUBIC).astype('uint8'),
        rih,riw,_ = ref_image.shape
        image = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2GRAY)
        ref_image = cv2.cvtColor(np.copy(ref_image), cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(max_matches, WTA_K=2)
        image_keypoints, image_descriptors = orb.detectAndCompute(image, None)
        ref_image_keypoints, ref_image_descriptors = orb.detectAndCompute(ref_image, None)
        matches = self.get_descriptor_matches(image_descriptors, ref_image_descriptors)
        num_matches = len(matches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = image_keypoints[match.queryIdx].pt
            points2[i, :] = ref_image_keypoints[match.trainIdx].pt

        if len(points1) >= 4:
            h_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

            if isinstance(h_matrix, np.ndarray) and np.count_nonzero(mask) >= 4:
                orb_image = cv2.warpPerspective(image, h_matrix, (riw, riw))
                return True, h_matrix, num_matches, orb_image

        return False, np.array([]), num_matches, None