__author__ = "Mario Bijelic"
__contact__ = "mario.bijelic@t-online.de"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import linregress

try:

    from utils import plot_2d_hist
    from planes import calculate_plane
    from phy_equations import total_transmittance_from_ground

except ModuleNotFoundError:

    from tools.wet_ground.utils import plot_2d_hist
    from tools.wet_ground.planes import calculate_plane
    from tools.wet_ground.phy_equations import total_transmittance_from_ground



def ground_water_augmentation(pointcloud, water_height=0.001, pavement_depth=0.0012, noise_floor=0.7, power_factor=15,
                              estimation_method='linear', flat_earth=False, debug=True,
                              delta=0.5, replace=True):
    """
    :param pointcloud:        5 column DENSE pointcloud format
    :param water_height:      highway aquaplaning > 2.5mm in [m]
    # https://www.gov.uk/aaib-reports/aar-1-2009-boeing-737-81q-g-xlac-avions-de-transport-regional-atr-72-202-g-bwda-and-embraer-emb-145eu-g-embo-29-december-2006 aquaplaning depth for airplanes
    :param pavement_depth:    fine graded 0.4-0.6, coarse graded 0.6-1.2, open graded friction course 1.5-3.5 mm
    # https://www.researchgate.net/publication/245283455_NJTxtr_-_A_computer_program_based_on_LASER_to_monitor_asphalt_segregation?enrichId=rgreq-dc45b0cf243aa1c384fa68cc033e1a4a-XXX&enrichSource=Y292ZXJQYWdlOzI0NTI4MzQ1NTtBUzoyMTc3ODg0MTg0MDAyNTZAMTQyODkzNjIxMzYwMw%3D%3D&el=1_x_3&_esc=publicationCoverPdf
    :param noise_floor        Assumed minimum percentage of estimated Intensity values from minimum ground reflectance values
    :param estimation_method  Define how to fit estimated laser parameters choice from linear and poly
    :param flat_earth         Define if to use flat earth assumption for incident angle calculation
    :param debug              Enable debug for debugging plots
    :return:                  augmented_pointcloud
    """

    w, h = calculate_plane(pointcloud)

    # Filter points below image plane
    height_over_ground = np.matmul(pointcloud[:, :3], np.asarray(w))
    height_over_ground = height_over_ground.reshape((len(height_over_ground), 1))
    ground = np.logical_and(np.matmul(pointcloud[:, :3], np.asarray(w)) + h < delta,
                            np.matmul(pointcloud[:, :3], np.asarray(w)) + h > -delta)

    ground_idx = np.where(ground)
    pointcloud_planes = np.hstack((pointcloud[ground, :], height_over_ground[ground]))
    if pointcloud_planes.shape[0] < 1000:
        return pointcloud
    if not flat_earth:
        # incident angle based on ground plane
        # calculate incident angle based on scalar product -> resulting in the cosine of the
        calculated_indicent_angle = np.arccos(np.divide(np.matmul(pointcloud_planes[:, :3], np.asarray(w)),
                                                        np.linalg.norm(pointcloud_planes[:, :3],
                                                                       axis=1) * np.linalg.norm(w)))
    elif flat_earth:
        # incident angle based on flat earth assumption
        calculated_indicent_angle = np.arccos(-np.divide(np.matmul(pointcloud_planes[:, :3], np.asarray([0, 0, 1])),
                                                         np.linalg.norm(pointcloud_planes[:, :3],
                                                                        axis=1) * np.linalg.norm([0, 0, 1])))
    else:
        assert False, 'flat earth tag has be bool'

    if debug:
        # flat earth assumption based on internal coordinate system
        plt.plot(np.linalg.norm(pointcloud_planes[:, :2], axis=1), calculated_indicent_angle * 180 / np.pi, 'x')
        plt.title('Incident Angles')
        plt.ylabel('Calculated incdent angle')
        plt.xlabel('distance')
        plt.legend(['Calculated Plane'])
        plt.show()
        plt.plot(np.linalg.norm(pointcloud_planes[:, :2], axis=1), pointcloud_planes[:, 3], 'x')
        plt.title('Intensitites')
        plt.ylabel('Intensity')
        plt.xlabel('distance')
        plt.show()

    relative_output_intensity, adaptive_noise_threshold, _, _ = estimate_laser_parameters(pointcloud_planes,
                                                                                          calculated_indicent_angle,
                                                                                          noise_floor=noise_floor,
                                                                                          estimation_method=estimation_method,
                                                                                          power_factor=power_factor,
                                                                                          debug=debug)

    # If average asphalt relfecitivity is 10% then we can use it to caclculate emmitted laserposer

    reflectivities = pointcloud_planes[:, 3] / np.cos(calculated_indicent_angle) / relative_output_intensity

    if debug:
        plot_2d_hist(np.linalg.norm(pointcloud_planes[:, :2], axis=1), reflectivities)

        plt.plot(np.linalg.norm(pointcloud_planes[:, :2], axis=1), reflectivities, 'x')
        plt.title('reflectivities')
        plt.ylabel('Reflectivity')
        plt.xlabel('distance')
        plt.show()
        plt.hist2d(np.linalg.norm(pointcloud_planes[:, :2], axis=1), reflectivities, bins=100)
        plt.title('reflectivities distance hist')
        plt.ylabel('Reflectivity')
        plt.xlabel('distance')
        plt.show()


    # assumes a minimum ground reflectivity of 5%
    rs, ts, rp, tp, aaout = total_transmittance_from_ground(calculated_indicent_angle,
                                                            rho=np.clip(reflectivities, 0.05, 1))

    # estimate noiselevel from pointcloud by taking minimum reflected point value
    if debug:
        plt.plot(np.linalg.norm(pointcloud_planes[:, :3], axis=1), np.maximum(tp, ts) / calculated_indicent_angle, 'x')
        plt.title('maximum transmittance/incident angle')
        plt.ylabel('Reflectivity')
        plt.xlabel('distance')
        plt.show()

    t = np.maximum(tp, ts)

    # weight relflection between clear and wet -> Assumes a 45° thread profile
    f = np.clip(water_height / pavement_depth, 0, 1)
    tw = (1 - f) * reflectivities + f * t / calculated_indicent_angle


    new_intensities = np.clip(relative_output_intensity * np.cos(calculated_indicent_angle) * tw, 0,
                              pointcloud_planes[:, 3])
    zero_points = new_intensities < (adaptive_noise_threshold * np.cos(calculated_indicent_angle))
    water_only_intensities = np.clip(t * relative_output_intensity, 0, 255)

    new_intensities[zero_points] = 0
    if debug:
        plt.plot(np.linalg.norm(pointcloud_planes[:, :3], axis=1), pointcloud_planes[:, 3], 'bx')
        plt.plot(np.linalg.norm(pointcloud_planes[:, :3], axis=1), new_intensities, 'yx')
        plt.plot(np.linalg.norm(pointcloud_planes[:, :3], axis=1), water_only_intensities, 'gx')
        plt.plot(np.linalg.norm(pointcloud_planes[:, :3], axis=1),
                 adaptive_noise_threshold * np.cos(calculated_indicent_angle), 'rx')
        plt.title('Overlayed intensitites')
        plt.ylabel('Intensities')
        plt.xlabel('Total Distance')
        plt.legend(['Original Values', 'New Intensities', 'Water Intensitites', 'Threshold'])
        plt.show()


    # Calculate which points to keep according to estimated adaptive noise threshold
    keep_points = new_intensities > adaptive_noise_threshold * np.cos(calculated_indicent_angle)
    keep_points_idx = np.where(keep_points)
    pointcloud_planes = pointcloud_planes[:, :5]

    augmented_pointcloud = np.zeros((pointcloud.shape[0] - ground_idx[0].shape[0] + keep_points_idx[0].shape[0], 5))
    augmented_pointcloud[:pointcloud.shape[0] - ground_idx[0].shape[0], :] = pointcloud[np.logical_not(ground), :]
    augmented_pointcloud[pointcloud.shape[0] - ground_idx[0].shape[0]:, :] = pointcloud_planes[keep_points_idx]
    augmented_pointcloud[pointcloud.shape[0] - ground_idx[0].shape[0]:, 3] = new_intensities[keep_points_idx]

    if replace:
        augmented_pointcloud[:, 4] = 0

    # save augmented flag into laser counter
    augmented_pointcloud[pointcloud.shape[0] - ground_idx[0].shape[0]:, 4] = 1

    return augmented_pointcloud


def filter_below_ground(pointcloud, w, h):
    above_ground = np.matmul(pointcloud[:, :3], np.asarray(w)) + h < 0.5
    pointcloud = pointcloud[above_ground, :]

    return pointcloud


def ransac_polyfit(x, y, order=3, n=15, k=100, t=0.1, d=15, f=0.8):
    # Applied https://en.wikipedia.org/wiki/Random_sample_consensus
    # Taken from https://gist.github.com/geohot/9743ad59598daf61155bf0d43a10838c
    # n – minimum number of data points required to fit the model
    # k – maximum number of iterations allowed in the algorithm
    # t – threshold value to determine when a data point fits a model
    # d – number of close data points required to assert that a model fits well to data
    # f – fraction of close data points required

    bestfit = np.polyfit(x, y, order)
    besterr = np.sum(np.abs(np.polyval(bestfit, x) - y))
    for kk in range(k):
        maybeinliers = np.random.randint(len(x), size=n)
        maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
        alsoinliers = np.abs(np.polyval(maybemodel, x) - y) < t
        if sum(alsoinliers) > d and sum(alsoinliers) > len(x) * f:
            bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
            thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers]) - y[alsoinliers]))
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
    return bestfit


def estimate_laser_parameters(pointcloud_planes, calculated_indicent_angle, power_factor=15, noise_floor=0.7,
                              debug=True, estimation_method='linear'):
    """
    :param pointcloud_planes: Get all points which correspond to the ground
    :param calculated_indicent_angle: The calculated incident angle for each individual point
    :param power_factor: Determines, how much more Power is available compared to a groundplane reflection.
    :param noise_floor: What are the minimum intensities that could be registered
    :param debug: Show additional Method
    :param estimation_method: Method to fit to outputted laser power.
    :return: Fits the laser outputted power level and noiselevel for each point based on the assumed ground floor reflectivities.
    """
    # normalize intensitities
    normalized_intensitites = pointcloud_planes[:, 3] / np.cos(calculated_indicent_angle)
    distance = np.linalg.norm(pointcloud_planes[:, :3], axis=1)

    # linear model
    p = None
    stat_values = None
    if len(normalized_intensitites) < 3:
        return None, None, None, None
    if estimation_method == 'linear':
        reg = linregress(distance, normalized_intensitites)
        w = reg[0]
        h = reg[1]
        p = [w, h]
        stat_values = reg[2:]
        relative_output_intensity = power_factor * (p[0] * distance + p[1])

    elif estimation_method == 'poly':
        # polynomial 2degre fit
        p = np.polyfit(np.linalg.norm(pointcloud_planes[:, :3], axis=1),
                       normalized_intensitites, 2)
        relative_output_intensity = power_factor * (
                p[0] * distance ** 2 + p[1] * distance + p[2])


    # estimate minimum noise level therefore get minimum reflected intensitites
    hist, xedges, yedges = np.histogram2d(distance, normalized_intensitites, bins=(50, 2555),
                                          range=((10, 70), (5, np.abs(np.max(normalized_intensitites)))))
    idx = np.where(hist == 0)
    hist[idx] = len(pointcloud_planes)
    ymins = np.argpartition(hist, 2, axis=1)[:, 0]
    min_vals = yedges[ymins]
    idx = np.where(min_vals > 5)
    min_vals = min_vals[idx]
    idx1 = [i + 1 for i in idx]
    x = (xedges[idx] + xedges[idx1]) / 2

    if estimation_method == 'poly':
        pmin = ransac_polyfit(x, min_vals, order=2)
        adaptive_noise_threshold = noise_floor * (
                pmin[0] * distance ** 2 + pmin[1] * distance + pmin[2])
    elif estimation_method == 'linear':
        if len(min_vals) > 3:
            pmin = linregress(x, min_vals)
        else:
            pmin = p
        adaptive_noise_threshold = noise_floor * (
                pmin[0] * distance + pmin[1])
    # Guess that noise level should be half the road relfection

    if debug:
        plt.plot(distance, normalized_intensitites, 'x')
        plt.plot(distance, relative_output_intensity, 'x')
        plt.plot(distance, adaptive_noise_threshold, 'x')
        plt.title('Estimated Lidar Parameters')
        plt.ylabel('Intensity')
        plt.xlabel('distance')
        plt.legend(['Input Intensities', 'Total Power', 'Noise Level'])
        plt.show()

    return relative_output_intensity, adaptive_noise_threshold, p, stat_values


def get_ground_plane_intensity_stats(pointcloud, recording=None, road_wettness=0, illustreate=True):
    w, h = calculate_plane(pointcloud)
    # Filter points below image plane
    height_over_ground = np.matmul(pointcloud[:, :3], np.asarray(w))
    height_over_ground = height_over_ground.reshape((len(height_over_ground), 1))
    ground = np.logical_and(np.matmul(pointcloud[:, :3], np.asarray(w)) + h < 0.3,
                            np.matmul(pointcloud[:, :3], np.asarray(w)) + h > -0.3)

    if len(ground)<1000:
        return None, None, None, None, None
    # Filter ground in vehicle trajectory
    ground = np.logical_and(ground, (pointcloud[:, 1] > -1.5) & (pointcloud[:, 1] < 1.5) & (pointcloud[:, 3] < 200))

    ground_idx = np.where(ground)
    pointcloud_planes = np.hstack((pointcloud[ground, :], height_over_ground[ground]))

    # incident angle based on ground plane
    calculated_indicent_angle = np.arccos(np.divide(np.matmul(pointcloud_planes[:, :3], np.asarray(w)),
                                                    np.linalg.norm(pointcloud_planes[:, :3], axis=1) * np.linalg.norm(
                                                        w)))

    distance = np.linalg.norm(pointcloud_planes[:, :3], axis=1)

    relative_output_intensity, adaptive_noise_threshold, p, stat_values = estimate_laser_parameters(pointcloud_planes,
                                                                                                    calculated_indicent_angle,
                                                                                                    debug=illustreate)
    if relative_output_intensity is None:
        return p, None, None, None, stat_values

    hist, xedges, yedges = np.histogram2d(distance, pointcloud_planes[:, 3], bins=(50, 255), range=((10, 70), (0, 255)))
    x = (xedges[:-1] + xedges[1:]) / 2
    y = (yedges[:-1] + yedges[1:]) / 2

    # get sums for each distance
    sums = np.sum(hist, axis=1)
    # filte not filled positions
    filled_idx = np.where(sums > 0)
    x = x[filled_idx]

    sums = sums[filled_idx]
    hist = hist[filled_idx[0], :]

    # filter each distance without elements

    # normalize hist along distance to get probs
    histp = hist / sums[:, None]
    # calculate comulative sum
    cumsum = np.cumsum(histp, axis=1)
    if illustreate:
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(x, y)

        z_min, z_max = np.abs(cumsum).min(), np.abs(cumsum).max()

        c = ax.pcolormesh(X, Y, cumsum.T, cmap='RdBu', vmin=z_min, vmax=z_max)
        if recording is not None:
            ax.set_title(recording + ' wettness =' + str(road_wettness))
        else:
            ax.set_title('pcolormesh')
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(c, ax=ax)

        plt.show()
    return p, x, histp, filled_idx, stat_values
