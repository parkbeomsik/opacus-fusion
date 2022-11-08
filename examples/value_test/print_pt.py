import torch

def print_pt(path):

    obj = torch.load(path)

    print(obj)

if __name__ == "__main__":
    print_pt("value_test/value_data/sample_conv_net_weight.pt")
            