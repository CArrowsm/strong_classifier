from argparse import ArgumentParser


### EDIT THESE PATHS ###
img_path = "/cluster/projects/bhklab/RADCURE/img/"
img_suffix = "_img.npy"                 # This string follows the patient ID in the filename
label_path = "/cluster/home/carrowsm/logs/label/artifact_labels.csv"
log_dir = None
### ---------------- ###

parser = ArgumentParser()
parser.add_argument("--img_dir", default=img_path, type=str)
parser.add_argument("--img_suffix", default=img_suffix, type=str)
parser.add_argument("--cal_acc", default=False, type=bool,
                    help='Whether or not to calculate the accuracy of predictions, based on image labels.')
parser.add_argument("--label_dir", default=None, type=str, help='Path to a CSV containing image labels.')
parser.add_argument("--logging", default=False, type=bool, help='Whether or not to save results.')
parser.add_argument("--logdir", default=log_dir, type=str, help='Where to save results.')

parse.add_argument("--norm_lower", default=0.64, type=float)
parse.add_argument("--norm_upper", default=0.86, type=float)


def get_args():
    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 1:
        logger.info("Unparsed args: {}".format(unparsed))

    d = vars(args)

    d["log_dir"] = LOGS_DIR

    return args, unparsed
