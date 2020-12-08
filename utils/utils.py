import mrcfile
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, distance_transform_cdt, grey_closing
from math import pi as PI
from lxml import etree
from skimage.measure import label

def read_mrc(filepath):
    """
    This function reads a mrc file
    
    Input
    ----------
        filepath: string
            The path of the mrc file
    Returns
    -------
        data: numpy array
            The content of the mrc file
    """
    with mrcfile.open(filepath, permissive=True) as mrc:
        data = mrc.data 
   
    return data

def write_mrc(data, filepath):
    """
    This function writes data to a mrc file
    
    Input
    ----------
        data: numpy array
            The data for the mrc file
        filepath: string
            The path of the mrc file
    Returns
    -------
    """
    with mrcfile.new(filepath, overwrite=True) as mrc:
        mrc.set_data(data)

def write_txt(coords, filepath, voxel_size = 14.08):
    """
    This function writes the particle centers' coordinates to a txt file.
    The generated txt file has 6 columes (PositionX,PositionY,PositionZ,VolumeX,VolumeY,VolumeZ) and each row denotes one particle.
     
    Input
    ----------
        coords: 2d list, e.g[[Xcoords1, Ycoords1, Zcoords1], [Xcoords2, Ycoords2, Zcoords2]...]
            A list includes XYZ coordinates of all particles
        filepath: string
            The path of the genrated txt file
        voxel_size: float, default 14.08 for spinach data
            The voxel size corresponding to data species
    Returns
    -----
    """
    with open(filepath, "w") as txtfile:
        print("PositionX    PositionY    PositionZ    VolumeX    VolumeY    VolumeZ", file=txtfile)
        for i in range(coords.shape[0]):
            volumeZ,volumeY,volumeX = coords[i]
            volumeZ,volumeY,volumeX = int(volumeZ), int(volumeY), int(volumeX)
            PositionZ,PositionY,PositionX = voxel_size*volumeZ, voxel_size*volumeY, voxel_size*volumeX
            print(round(PositionX,2),round(PositionY,2),round(PositionZ,2),volumeX,volumeY,volumeZ, file=txtfile) 

def read_xml(filename):
    """
    This function reads the particle information from a xml file. Copied from https://gitlab.inria.fr/serpico/deep-finder/-/blob/master/deepfinder/utils/objl.py#L67
     
    Input
    ----------
        filename: string
            The path of the xml file
    Returns
    -----
        objlOUT: list of dictionary
            A list containing dictionaries, which includes the information of each particle
    """
    tree = etree.parse(filename)
    objl_xml = tree.getroot()

    objlOUT = []
    for p in range(len(objl_xml)):
        tidx  = objl_xml[p].get('tomo_idx')
        objid = objl_xml[p].get('obj_id')
        lbl   = objl_xml[p].get('class_label')
        x     = objl_xml[p].get('x')
        y     = objl_xml[p].get('y')
        z     = objl_xml[p].get('z')
        psi   = objl_xml[p].get('psi')
        phi   = objl_xml[p].get('phi')
        the   = objl_xml[p].get('the')
        csize = objl_xml[p].get('cluster_size')

        # if facultative attributes exist, then cast to correct type:
        if tidx!=None:
            tidx = int(tidx)
        if objid!=None:
            objid = int(objid)
        if csize!=None:
            csize = int(csize)
        if psi!=None or phi!=None or the!=None:
            psi = float(psi)
            phi = float(phi)
            the = float(the)
        
        obj = {
        'tomo_idx':tidx,
        'obj_id'  :objid,
        'label'   :int(lbl),
        'x'       :float(x) ,
        'y'       :float(y) ,
        'z'       :float(z) ,
        'psi'     :psi,
        'phi'     :phi,
        'the'     :the,
        'cluster_size':csize}
        
        objlOUT.append(obj)
        
    return objlOUT



def load_data(path_data, path_target):
    """
    This function loads data (.mrc) and correspnding targets(.mrc). Modified from https://gitlab.inria.fr/serpico/deep-finder/-/blob/master/deepfinder/utils/core.py#L122.
     
    Input
    ----------
        path_data: list of string, e.g. [path_data1,path_data2...]
            A list includes paths of training data
        path_target: list of string, e.g. [path_target1,path_target2...]
            A list includes paths of training groundtruth, the order of targets should correpond to the order of data
    Returns
    -----
        data_list: list of array
            A list includes data
        target_list: list of array
            A list includes targets
    """
    data_list   = []
    target_list = []
    for idx in range(0,len(path_data)):
        data   = read_mrc(path_data[idx])
        target = read_mrc(path_target[idx])
        
        assert data.shape == target.shape, 'Tomogram and target are not of same size!'
            
        data_list.append(data)
        target_list.append(target)
        
    return data_list, target_list

def get_patch_position(tomodim, p_in, obj, Lrnd):
    '''
    Takes position specified in 'obj', applies random shift to it, and then checks if the patch around this position is out of the tomogram boundaries. If so, the position is shifted to that patch is inside the tomo boundaries. Modified from https://gitlab.inria.fr/serpico/deep-finder/-/blob/master/deepfinder/utils/core.py#L171.
    
    Input
    ----------
        tomodim: tuple
            (dimX,dimY,dimZ), size of tomogram
        p_in: int 
            lenght of patch in voxels
        obj: dictionary 
            obtained when calling objlist[idx]
        Lrnd: int 
            random shift in voxels applied to position
        
    Returns
    -----
        x,y,z  : int,int,int 
            coordinates for sampling patch safely
    '''
    # sample at coordinates specified in obj=objlist[idx]
    x = int( obj['x'] )
    y = int( obj['y'] )
    z = int( obj['z'] )
        
    # Add random shift to coordinates:
    x = x + np.random.choice(range(-Lrnd,Lrnd+1))
    y = y + np.random.choice(range(-Lrnd,Lrnd+1))
    z = z + np.random.choice(range(-Lrnd,Lrnd+1))
    
    # Shift position if too close to border:
    if (x<p_in) : x = p_in
    if (y<p_in) : y = p_in
    if (z<p_in) : z = p_in
    if (x>tomodim[2]-p_in): x = tomodim[2]-p_in
    if (y>tomodim[1]-p_in): y = tomodim[1]-p_in
    if (z>tomodim[0]-p_in): z = tomodim[0]-p_in
    
    return x,y,z

def dist_label(label, neighbor_radius=None, apply_grayscale_closing=True):
    """ 
    Cell center label creation (Euclidean distance). Modified from https://bitbucket.org/t_scherr/cell-segmentation-and-tracking/src/master/segmentation/training/train_data_representations.py#lines-167

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param neighbor_radius: Defines the area to look for neighbors (smaller radius in px decreases the computation time)
        :type neighbor_radius: int
    :param apply_grayscale_closing: close gaps in between neighbor labels.
        :type apply_grayscale_closing: bool
    :return: Cell distance label image, neighbor distance label image.
    """
    # Relabel label to avoid some errors/bugs
    label_dist = np.zeros(shape=label.shape, dtype=np.float)
    label_dist_neighbor = np.zeros(shape=label.shape, dtype=np.float)

    mean_diameter = []
    for i in np.unique(label):
        mean_diameter.append((6 * np.sum(label==i) / PI) ** (1 / 3))
    mean_diameter = np.mean(np.array(mean_diameter))
    neighbor_radius = 3 * mean_diameter

    # Find centroids, crop image, calculate distance transform
    for i in np.unique(label)[1:]:

        # Get nucleus and Euclidean distance transform for each nucleus
        nucleus = (label == i)
        centroid, diameter = np.mean(np.where(nucleus),axis=-1), ((6 * np.sum(nucleus) / PI) ** (1 / 3))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
                       int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
                       int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
                       ].astype(np.float32)
        nucleus_crop_dist = distance_transform_edt(nucleus_crop)
        nucleus_crop_dist[nucleus_crop_dist==1] = 0
        if np.max(nucleus_crop_dist) > 0:
            nucleus_crop_dist = nucleus_crop_dist / np.max(nucleus_crop_dist)

        label_dist[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
        int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
        ] += nucleus_crop_dist

        # Get crop containing neighboring nuclei
        nucleus_neighbor_crop = np.copy(label[
                                int(max(centroid[0] - neighbor_radius, 0)):int(
                                    min(centroid[0] + neighbor_radius, label.shape[0])),
                                int(max(centroid[1] - neighbor_radius, 0)):int(
                                    min(centroid[1] + neighbor_radius, label.shape[1])),
                                int(max(centroid[2] - neighbor_radius, 0)):int(
                                    min(centroid[2] + neighbor_radius, label.shape[2]))
                                ])

    return label_dist