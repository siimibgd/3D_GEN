import depth_estimation
import fuse_depth_maps
import point_cloud_generation

def main():

    depth_estimation.run()
    fuse_depth_maps.run()
    point_cloud_generation.run()

if __name__=="__main__":
    main()