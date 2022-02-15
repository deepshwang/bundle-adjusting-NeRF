from data.static_blender import Dataset
import options
import os,sys,time


if __name__ == "__main__":
    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)

    dataset = Dataset(opt, split="train")
    out = dataset[1]
    print("leggo")



