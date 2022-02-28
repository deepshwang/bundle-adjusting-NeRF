from data.static_blender import Dataset
import options
import os,sys,time

class A():
    def __init__(self):
        self.attr_a = ''
        self.operation()

    def operation(self):
        self.attr_a += 'a'

class B(A):
    def __init__(self):
        super().__init__()

    def operation(self):
        self.attr_a += 'b'

if __name__ == "__main__":
    inst = B()
    print(inst.attr_a)









    # opt_cmd = options.parse_arguments(sys.argv[1:])
    # opt = options.set(opt_cmd=opt_cmd)
    #
    # dataset = Dataset(opt, split="train")
    # out = dataset[1]
    # print("leggo")



